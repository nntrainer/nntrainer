// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor.h
 * @date	01 December 2023
 * @brief	This is a Tensor class
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__
#ifdef __cplusplus

#define MAKE_SHARED_TENSOR(...) std::make_shared<nntrainer::Tensor>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS(tensor, ...)                                      \
  do {                                                                         \
    if (tensor.empty())                                                        \
      tensor = Tensor(__VA_ARGS__);                                            \
  } while (0);

#include <cstddef>

#include <compute_ops.h>
#include <nntrainer_log.h>
#include <tensor_base.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif

namespace nntrainer {

class LazyTensor;

/**
 * @class Tensor Class
 * @brief Tensor is a multidimensional matrix that contain elements of a single
 * data type and can perform various operations like addition, division,
 * multiplication, dot product, data averaging, and more.
 * NNTrainer defines tensor types using different data types and memory formats.
 * Supported data types and format are specified in the file 'tensor_dim.h'.
 *
 * @note The Tensor class utilizes the TensorBase class to support tensors with
 * various data types. In other words, this tensor class serves as a container
 * for tensors, and thus the functionality of the tensor should be defined in
 * each tensor class (FloatTensor, HalfTensor, etc.).
 *
 */
class Tensor {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32);

  /**
   * @brief     Constructor of Tensor with dimension, possibly lazily
   * @param d Tensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   * @param qscheme_ Quantization scheme (only applies to Quantized Tensor)
   * @param is_virtual virtual tensor boolean (default=false)
   */
  Tensor(const TensorDim &d, bool alloc_now,
         Initializer init = Initializer::NONE, std::string name = "",
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE,
         bool is_virtual = false);

  /**
   * @brief     Constructor of Tensor with dimension/buf
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @param qscheme_ Quantization scheme (only applies to Quantized Tensor)
   * @note Memory for this tensor is instantaneously allocated
   */
  Tensor(const TensorDim &d, const void *buf = nullptr,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE);

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d0, size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(TensorDim(d0, d1, d2, d3, fm, d_type), nullptr, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, d1, d2, d3, fm, d_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, 1, d2, d3, fm, d_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  explicit Tensor(size_t d3, Tformat fm = Tformat::NCHW,
                  Tdatatype d_type = Tdatatype::FP32,
                  QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, 1, 1, d3, fm, d_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d0, size_t d1, size_t d2, size_t d3,
         ml::train::TensorDim::TensorType t_type,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(TensorDim(d0, d1, d2, d3, t_type), nullptr, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d1, size_t d2, size_t d3,
         ml::train::TensorDim::TensorType t_type,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, d1, d2, d3, t_type){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(size_t d2, size_t d3, ml::train::TensorDim::TensorType t_type,
         QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3,
           (t_type.format == Tformat::NCHW) ? d2 : 1,
           (t_type.format == Tformat::NCHW) ? d3 : d2, t_type, qscheme_){};
  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  explicit Tensor(size_t d3, ml::train::TensorDim::TensorType t_type,
                  QScheme qscheme_ = QScheme::PER_TENSOR_AFFINE) :
    Tensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3, 1,
           (t_type.format == Tformat::NCHW) ? d3 : 1, t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
         ml::train::TensorDim::TensorType t_type);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<float>>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<float>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#ifdef ENABLE_FP16
  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   * @todo      It is more desirable to move this implementaton into
   *            `tensor.cpp`, for it requires half_tensor.h
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
         ml::train::TensorDim::TensorType t_type) {
    itensor_ = std::make_unique<HalfTensor>(d, t_type.format);
  }

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<_FP16>>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<_FP16>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};
#endif

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<uint8_t>>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<uint8_t>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<uint8_t>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<uint16_t>>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<uint16_t>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<uint16_t>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<uint32_t>>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<std::vector<uint32_t>>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  Tensor(std::vector<std::vector<uint32_t>> const &d,
         std::vector<float> const &scales,
         std::vector<unsigned int> const &zero_points,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, zero_points,
           t_type, qscheme_){};

  /**
   * @brief     Constructor of CharTensor (QINT8)
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_);

  /**
   * @brief     Constructor of CharTensor (QINT8)
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<std::vector<int8_t>>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, t_type,
           qscheme_){};

  /**
   * @brief     Constructor of CharTensor (QINT8)
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<int8_t>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, t_type,
           qscheme_){};

  /**
   * @brief     Constructor of CharTensor (QINT16)
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<int16_t>>>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_);

  /**
   * @brief     Constructor of CharTensor (QINT16)
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<std::vector<int16_t>>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, t_type,
           qscheme_){};

  /**
   * @brief     Constructor of CharTensor (QINT16)
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] scales scale factors for the Tensor.
   * @param[in] t_type Tensor Type
   * @param[in] qscheme_ Quantization scheme (only applies to Quantized Tensor)
   */
  Tensor(std::vector<std::vector<int16_t>> const &d,
         std::vector<float> const &scales,
         ml::train::TensorDim::TensorType t_type, QScheme qscheme_) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, scales, t_type,
           qscheme_){};

  /**
   *  @brief  Constructor of Tensor by directly assigning TensorBase.
   *  @param[in] rhs unique_ptr of a TensorBase
   *  @note TensorBase is an abstract class so we can't directly instantiate
   it.
   *  Make sure to use a unique_ptr with a derived class when utilizing this
   *  constructor.
   */
  Tensor(const std::unique_ptr<TensorBase> &rhs);

  /**
   * @brief Basic Destructor
   */
  ~Tensor() noexcept {
    if (is_virtual && mapped_ptr != nullptr) {
      deactivate();
    }
  };

  /**
   *  @brief  Copy constructor of Tensor.
   *  @param[in] Tensor &
   */
  Tensor(const Tensor &rhs);

  /**
   *  @brief  Move constructor of Tensor.
   *  @param[in] Tensor &&
   */
  Tensor(Tensor &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs Tensor to be copied.
   */
  Tensor &operator=(const Tensor &rhs);

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Tensor to be moved.
   */
  Tensor &operator=(Tensor &&rhs) noexcept = default;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const Tensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const Tensor &rhs) const { return !(*this == rhs); }

  /**
   *  @brief  Compare itensor considering dynamic type checking.
   *  @param[in] lhs pointer of a TensorBase
   *  @param[in] rhs pointer of a TensorBase
   */
  template <typename T>
  static bool itensorCompare(const TensorBase *lhs, const TensorBase *rhs) {
    auto lhs_cast = dynamic_cast<const T *>(lhs);
    auto rhs_cast = dynamic_cast<const T *>(rhs);

    if (!lhs_cast || !rhs_cast) {
      return false;
    }

    return *lhs_cast == *rhs_cast;
  }

  /**
   * @brief Construct a new Tensor object from a buffer
   * This will not copy buffer to a new tensor but directly uses it
   *
   * @param[in] buf buffer
   * @param[in] bytes buffer size in bytes
   * @param[in] d tensor dim
   * @param[in] offset offset to be used from current
   * @return    Tensor object
   * @throws    std::invalid_argument if buf is null
   */
  template <typename T = float>
  static Tensor Map(T *buf, unsigned int bytes, const TensorDim &d,
                    size_t offset = 0) {
    if (d.getDataLen() == 0 || buf == nullptr) {
      throw std::invalid_argument(
        "[Tensor::Map] empty tensor dim is not allowed");
    }

    if (d.getDataLen() * sizeof(T) + offset > bytes) {
      throw std::invalid_argument(
        "Creating shared tensor of size bigger than tensor memory.");
    }

    Tensor output("", d.getFormat(), d.getDataType());
    output.setTensorVar(d, buf, offset);
    return output;
  };

  /**
   * @brief    Allocate memory for this tensor
   */
  void allocate();

  /**
   * @brief    Deallocate memory for this tensor
   * @note     This will not necessary free the memory as tensors share memory
   */
  void deallocate();

  /**
   * @brief    Check if the tensor has memory allocated/assigned/associated
   */
  bool isAllocated();

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer
   */
  template <typename T = float> T *getData() const {
    return (T *)itensor_->getData();
  }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer
   */
  template <typename T = float> T *getData(size_t idx) const {
    return (T *)itensor_->getData(idx);
  }

  /**
   * @brief     return scale pointer of Tensor
   * @retval    template T pointer
   */
  template <typename T = float> T *getScale() const {
    return (T *)itensor_->getScale();
  }

  /**
   * @brief     return scale pointer of Tensor
   * @retval    template T pointer
   */
  template <typename T = float> T *getScale(size_t idx) const {
    return (T *)itensor_->getScale(idx);
  }

  /**
   * @brief     return zero point pointer of Tensor
   * @retval    unsigned int pointer
   */
  unsigned int *getZeroPoint() const { return itensor_->getZeroPoint(); }

  /**
   * @brief     return zero point pointer of Tensor
   * @retval    unsigned int pointer
   */
  unsigned int *getZeroPoint(size_t idx) const {
    return itensor_->getZeroPoint(idx);
  }

  /**
   * @brief     i data index
   * @retval    template T pointer (address of ith data)
   */
  template <typename T = float> T *getAddress(unsigned int i) {
    return (T *)itensor_->getAddress(i);
  }

  /**
   * @brief     i data index
   * @retval    template T pointer (address of ith data)
   */
  template <typename T = float> const T *getAddress(unsigned int i) const {
    return (T *)itensor_->getAddress(i);
  }

  /**
   * @brief    get address of n-d data
   */
  template <typename T = float>
  T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                unsigned int w) {
    return getAddress<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief    get address of n-d data
   */
  template <typename T = float>
  const T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w) const {
    return getAddress<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  template <typename T = float>
  const T &getValue(unsigned int idx) const noexcept {
    return getData<T>()[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  template <typename T = float> T &getValue(unsigned int idx) noexcept {
    return getData<T>()[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  template <typename T = float>
  const T &getValue(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w) const noexcept {
    return getValue<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  template <typename T = float>
  T &getValue(unsigned int b, unsigned int c, unsigned int h,
              unsigned int w) noexcept {
    return getValue<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief     Fill the Tensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Set the element value
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value);

  /**
   * @brief     Set the element value
   * @param[in] offset offset from start location
   * @param[in] value value to be stored
   *
   * @todo      This is a temporary workout. Remove this
   */
  void setValueInt(unsigned int offset, int value) noexcept {
    int *data_int = (int *)getData();
    data_int[offset] = value;
  }

  /**
   * @brief     add the element value to the location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   * @param[in] beta scalar to multiply output with and add
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) noexcept;

  /**
   * @brief     Fill the Tensor elements with zero
   */
  void setZero();

  /**
   * @brief     Set the tensor with random normal distribution
   * @param[in] mean mean of the distribution
   * @param[in] std standard deviation of the distribution
   */
  void setRandNormal(float mean = 0.0f, float stddev = 0.05f);

  /**
   * @brief     Set the tensor with random uniform distribution
   * @param[in] min minimum value for the distribution
   * @param[in] max maximum value for the distribution
   */
  void setRandUniform(float min = -0.05f, float max = 0.05f);

  /**
   * @brief     Set the tensor with random bernoulli distribution
   * @param[in] probability probability value for the distribution
   */
  void setRandBernoulli(float probability = 0.5f);

  /**
   * @brief     Initialize the memory of the given tensor
   */
  void initialize();

  /**
   * @brief     Initialize the memory of the given tensor
   * @param     init Initiailizer to use for the initialization
   */
  void initialize(Initializer init);

  /**
   * @brief Apply instantly to the element
   * @param[in] *function function pointer applied
   * @return int ML_ERROR_NONE if successful
   */
  template <typename T = float> int apply_i(std::function<T(T)> f) {
    Tensor result = *this;
    apply<T>(f, result);

    return ML_ERROR_NONE;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  template <typename T = float> Tensor apply(std::function<T(T)> f) const {
    Tensor result;
    apply<T>(f, result);

    return result;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  template <typename T = float>
  Tensor &apply(std::function<T(T)> f, Tensor &output) const {
    CREATE_IF_EMPTY_DIMS(output, itensor_->getDim(), nullptr);

    if (itensor_->getFormat() != output.itensor_->getFormat() ||
        itensor_->getDataType() != output.itensor_->getDataType()) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[Tensor::apply] output format or data type does not match");
    }

    itensor_->apply(f, output);

    return output;
  }

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor apply(std::function<Tensor(Tensor)> f) const;

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  Tensor &apply(std::function<Tensor &(Tensor, Tensor &)> f,
                Tensor &output) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(Tensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor multiply_strided(Tensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor &multiply_strided(Tensor const &m, Tensor &output,
                           const float beta = 0.0) const;

  /**
   * @brief     Multiply value element by element immediately
   * @param[in] value multiplier
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int multiply_i(float const &value, ComputeOps *ops = nullptr);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated Tensor
   */
  Tensor multiply(float const &value) const;

  /**
   * @brief      multiply value element by element
   * @param[in]  value multiplier
   * @param[out] out out tensor to store the result
   * @retval     Calculated Tensor
   */
  Tensor &multiply(float const &value, Tensor &out,
                   ComputeOps *ops = nullptr) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(Tensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   */
  Tensor multiply(Tensor const &m, const float beta = 0.0) const;

  /**
   * @brief      Multiply Tensor Element by Element ( Not the MxM )
   * @param[in]  m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in]  beta scalar to multiply output with and add
   * @retval     Calculated Tensor
   */
  Tensor &multiply(Tensor const &m, Tensor &output,
                   const float beta = 0.0, ComputeOps *ops = nullptr) const;

  /**
   * @brief     Divide value element by element immediately
   * @param[in] value divisor
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int divide_i(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated Tensor
   */
  Tensor divide(float const &value) const;

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  Tensor &divide(float const &value, Tensor &output,
                 ComputeOps *ops = nullptr) const;

  /**
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int divide_i(Tensor const &m);

  /**
   * @brief     Divide Tensor Element by Element
   * @param[in] m Divisor Tensor
   * @retval    Calculated Tensor
   */
  Tensor divide(Tensor const &m) const;

  /**
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  Tensor &divide(Tensor const &m, Tensor &output, ComputeOps *ops = nullptr) const;

  /**
   * @brief     Add Tensor Elementwise
   * @param[in] input Tensor to be added
   * @param[in] beta scalar to add output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add_i
   */
  int add_i_strided(Tensor const &input, const float beta = 0.0);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] input Tensor to be added
   * @param[in] beta Value to be scale the input tensor
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  Tensor add_strided(Tensor const &input, const float beta = 0.0) const;

  /**
   * @brief      Add Tensor Element by Element
   * @param[in]  input Tensor to be added
   * @param[out] output Tensor to store the result
   * @param[in]  beta Value to be scale the input tensor
   * @retval     Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  Tensor &add_strided(Tensor const &input, Tensor &output,
                      const float beta = 0.0) const;

  /**
   * @brief     Add Tensor Element immediately to target tensor without mem copy
   * @param[in] value value to be added
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(float const &value);

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Tensor
   */
  Tensor add(float const &value) const;

  /**
   * @brief      Add Tensor Element by Element
   * @param[in]  value value to be added
   * @param[out] output Tensor to save output without allocating new memory
   * @retval     Calculated Tensor
   */
  Tensor &add(float const &value, Tensor &output,
              ComputeOps *ops = nullptr) const;

  /**
   * @brief     Add Tensor Element by Element without mem copy
   * @param[in] m Tensor to be added
   * @param[in] alpha Values to be scaled
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(Tensor const &m, float const alpha = 1.F);

  /**
   * @brief Do add_i for specific section
   *
   * @param len Length of the specific section
   * @param addr_idx Starting index of the psecific section
   * @param m Input Tensor to be added
   * @param incX Incremental index of X
   * @param incY Incremental index of Y
   * @param alphas Vector of multiple alpha values
   * @param alpha_idx Index of alpha in alpha vector
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i_partial(unsigned int len, unsigned int addr_idx, Tensor &m,
                    unsigned int incX, unsigned int incY, const Tensor alphas,
                    unsigned int alpha_idx, ComputeOps *ops = nullptr);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[in] alpha Values to be scaled
   * @retval    Calculated Tensor
   */
  Tensor add(Tensor const &m, float const alpha = 1) const;

  /**
   * @brief      Add Tensor Element by Element
   * @param[in]  m Tensor to be added
   * @param[out] output Tensor to be out
   * @param[in]  alpha Values to be scaled
   * @retval     Calculated Tensor
   */
  Tensor &add(Tensor const &m, Tensor &output, float const alpha = 1, ComputeOps *ops = nullptr) const;

  /**
   * @brief     memcpyless version of subtract
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(float const &value);

  /**
   * @brief     subtract value Element by Element
   * @param[in] value value to be subtracted
   * @retval    Calculated Tensor
   */
  Tensor subtract(float const &value) const;

  /**
   * @brief      Subtract Tensor Element by Element
   * @param[in]  value value to be added
   * @param[out] output Tensor to save output without allocating new memory
   * @retval     Calculated Tensor
   */
  Tensor &subtract(float const &value, Tensor &output) const;

  /**
   * @brief     memcpyless version of subtract
   * @param[in] m Tensor to be subtracted
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(Tensor const &m);

  /**
   * @brief     Substract Tensor Element by Element
   * @param[in] m Tensor to be subtracted
   * @retval    Calculated Tensor
   */
  Tensor subtract(Tensor const &m) const;

  /**
   * @brief      Subtract Tensor Element by Element
   * @param[in]  m Tensor to be added
   * @param[out] output Tensor to be out
   * @retval     Calculated Tensor
   */
  Tensor &subtract(Tensor const &m, Tensor &output) const;

  /**
   * @brief     sum all the Tensor elements according to the batch
   * @retval    Calculated Tensor(batch, 1, 1, 1)
   */
  Tensor sum_by_batch(ComputeOps *ops = nullptr) const;

  /**
   * @brief     sum all the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated Tensor
   */
  Tensor sum(unsigned int axis, float alpha = 1.0) const;

  /**
   * @brief     sum all the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[out] output output tensor
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated Tensor
   */
  Tensor &sum(unsigned int axis, Tensor &output, float alpha = 1.0,
              float beta = 0.0) const;

  /**
   * @brief sum all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @param alpha Scale the sum by this value
   * @return Tensor
   */
  Tensor sum(const std::vector<unsigned int> &axes, float alpha = 1.0) const;

  /**
   * @brief sum all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @param[out] output output tensor
   * @param alpha Scale the sum by this value
   * @return Tensor
   */
  Tensor &sum(const std::vector<unsigned int> &axes, Tensor &output,
              float alpha = 1.0) const;

  /**
   * @brief  return absolute value
   * @retval Calculated Tensor
   */
  Tensor &abs(Tensor &output, ComputeOps *ops = nullptr) const;

  /**
   * @brief     Averaging the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @retval    Calculated Tensor
   */
  Tensor average(unsigned int axis) const;

  /**
   * @brief     Averaging the Tensor elements according to the axis
   * @retval    Calculated Tensor
   */
  Tensor &average(unsigned int axis, Tensor &output) const;

  /**
   * @brief     Average all the Tensor by multiple axes
   * @param[in] axes axes to sum along
   * @retval    Calculated Tensor
   */
  Tensor average(const std::vector<unsigned int> &axes) const;

  /**
   * @brief      Average all the Tensor by multiple axes
   * @param[in]  axes axes to sum along
   * @param[out] output output tensor
   * @retval     Calculated Tensor
   */
  Tensor &average(const std::vector<unsigned int> &axes, Tensor &output) const;

  /**
   * @brief     Average the Tensor elements by all axis
   * @retval    Calculated Tensor
   */
  Tensor average() const;

  /**
   * @brief     Averaging the Tensor elements by all axis
   * @retval    Calculated Tensor
   */
  Tensor &average(Tensor &output) const;

  /**
   * @brief     Tensor power element without mem copy
   * @param[in] exponent exponent
   * @retval    #ML_ERROR_NONE  Successful
   */
  int pow_i(float exponent);

  /**
   * @brief     Tensor power element by element
   * @param[in] exponent exponent
   * @retval    Calculated Tensor
   */
  Tensor pow(float exponent) const;

  /**
   * @brief      Tensor power element by element
   * @param[in]  exponent exponent
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &pow(float exponent, Tensor &output) const;

  /**
   * @brief     Compute square-root element by element
   * @retval    #ML_ERROR_NONE  Successful
   */
  int sqrt_i();

  /**
   * @brief     Compute square-root by element
   * @retval    Calculated Tensor
   */
  Tensor sqrt() const;

  /**
   * @brief      Compute square-root by element
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &sqrt(Tensor &output) const;

  /**
   * @brief     Compute negation by element
   * @retval    Calculated Tensor
   */
  Tensor neg() const;

  /**
   * @brief      Compute negation by element
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &neg(Tensor &output) const;

  /**
   * @brief     Gauss error function
   * @retval    #ML_ERROR_NONE  Successful
   */
  int erf_i();

  /**
   * @brief     Gauss error function
   * @retval    Calculated Tensor
   */
  Tensor erf() const;

  /**
   * @brief      Gauss error function
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &erf(Tensor &output) const;

  /**
   * @brief    sin transform function
   * @param[out] out out to store the result
   */
  void sin(Tensor &out, float alpha = 1.0) const;

  /**
   * @brief    cos transform function
   * @param[out] out out to store the result
   */
  void cos(Tensor &out, float alpha = 1.0) const;

  /**
   * @brief tangent transform function
   * @param[out] output out to store the result
   */
  void tan(Tensor &output, float alpha = 1.0) const;

  /**
   * @brief     Compute exponential element-wise (in-place)
   * @retval    #ML_ERROR_NONE  Successful
   */
  int exp_i();

  /**
   * @brief     Compute exponential element-wise
   * @retval    Calculated Tensor
   */
  Tensor exp() const;

  /**
   * @brief      Compute exponential element-wise
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &exp(Tensor &output) const;

  /**
   * @brief     Compute natural logarithm element-wise (in-place)
   * @retval    #ML_ERROR_NONE  Successful
   */
  int log_i();

  /**
   * @brief     Compute natural logarithm element-wise
   * @retval    Calculated Tensor
   */
  Tensor log() const;

  /**
   * @brief      Compute natural logarithm element-wise
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &log(Tensor &output) const;

  /**
   * @brief     Clamp tensor values to [min, max] (in-place)
   * @param[in] min minimum value
   * @param[in] max maximum value
   * @retval    #ML_ERROR_NONE  Successful
   */
  int clamp_i(float min, float max);

  /**
   * @brief     Clamp tensor values to [min, max]
   * @param[in] min minimum value
   * @param[in] max maximum value
   * @retval    Calculated Tensor
   */
  Tensor clamp(float min, float max) const;

  /**
   * @brief      Clamp tensor values to [min, max]
   * @param[in]  min minimum value
   * @param[in]  max maximum value
   * @param[out] output out to store the result
   * @retval     Calculated Tensor
   */
  Tensor &clamp(float min, float max, Tensor &output) const;

  /**
   * @brief inverse squared root function (in-place)
   */
  void inv_sqrt_i();

  /**
   * @brief inverse squared root function
   * @param[in] out output Tensor
   */
  Tensor inv_sqrt(Tensor &out, ComputeOps *ops = nullptr) const;

  /**
   * @brief     Anchor a starting point to defer following evaluation
   * @retval    LazyTensor class that can be used with run();
   */
  LazyTensor chain() const;

  /**
   * @brief     l2norm the Tensor elements
   * @retval    Calculated l2norm
   */
  float l2norm(ComputeOps *ops = nullptr) const;

  /**
   * @brief     Normalize the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor &normalization(Tensor &output) const;

  /**
   * @brief     Standardize the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor &standardization(Tensor &output) const;

  /**
   * @brief     Normalize the Tensor elements (in-place)
   * @note      min-max normalization
   */
  void normalization_i();

  /**
   * @brief     Normalize the Tensor along dimension (in-place)
   * @param[in] dim Dimension to normalize
   * @param[in] p Norm order (default 2.0)
   * @param[in] epsilon Epsilon for stability
   */
  void normalization_i(unsigned int dim, float p = 2.0, float epsilon = 1e-12);

  /**
   * @brief     Standardize the Tensor elements in-place
   * @retval    Calculated Tensor
   */
  void standardization_i();

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed input tensor.
   * @param[in] input Tensor
   * @param[in] trans Transpose
   * @param[in] trans_in Transpose input
   * @retval    Calculated Tensor
   */
  Tensor dot(Tensor const &input, bool trans = false,
             bool trans_in = false) const;

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and
   * second-last dimension of passed input tensor.
   * @param[in] input Tensor
   * @param[in] output output Tensor
   * @param[in] trans Transpose
   * @param[in] trans_in Transpose input
   * @param[in] beta beta
   * @retval    Calculated Tensor
   */
  Tensor &dot(Tensor const &input, Tensor &output, bool trans = false,
              bool trans_in = false, float beta = 0.0f,
              ComputeOps *ops = nullptr) const;

  void dot(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs,
           bool trans = false, bool trans_in = false, float beta = 0.0f) const;

  /**
   * @brief compute the derivative of this in the current tensor
   * @param input same as given to the dot()
   * @param output_deriv the derivative of the output
   * @param[in] trans same as given to the dot()
   * @param[in] trans_in same as given to the dot()
   * @param[in] beta same as given to the dot()
   * @note This will compute the derivative in-place and will overwrite
   existing
   * data in the tensor
   */
  Tensor &dot_deriv_wrt_1(Tensor const &input, Tensor const &output_deriv,
                          bool trans = false, bool trans_in = false,
                          float beta = 0.0f);

  /**
   * @brief compute the derivative wrt m in the input tensor
   * @param input_deriv tensor where derivative wrt m will be stored
   * @param output_deriv the derivative of the output
   * @param[in] trans same as given to the dot()
   * @param[in] trans_in same as given to the dot()
   * @param[in] beta same as given to the dot()
   * @note The caller tensor must be the same tensor as the one which called
   the dot() product.
   */
  Tensor &dot_deriv_wrt_2(Tensor &input_deriv, Tensor const &output_deriv,
                          bool trans = false, bool trans_in = false,
                          float beta = 0.0f) const;

  /**
   * @copydoc Tensor::dot(Tensor const &input, Tensor &output, bool trans,
              bool trans_in, float beta) const
   * @details performs dot operation over a batch of inputs. If the batch sizes
   of the given two tensors are different, the bigger one should be a multiple
   of the smaller one.
   */
  Tensor &dotBatched(Tensor const &input, Tensor &result, bool trans = false,
                     bool trans_in = false, float beta = 0.0f) const;

  /**
   * @copydoc Tensor::dot_deriv_wrt_1(Tensor const &input, Tensor const
   &output_deriv, bool trans, bool trans_in, float beta)
   */
  Tensor &dot_batched_deriv_wrt_1(Tensor const &input,
                                  Tensor const &output_deriv,
                                  bool trans = false, bool trans_in = false,
                                  float beta = 0.0f);

  /**
   * @brief Tensor::dot_deriv_wrt_2(Tensor const &input_deriv, Tensor const
   &output_deriv, bool trans, bool trans_in, float beta) const
   */
  Tensor &dot_batched_deriv_wrt_2(Tensor &input_deriv,
                                  Tensor const &output_deriv,
                                  bool trans = false, bool trans_in = false,
                                  float beta = 0.0f) const;

  /**
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate)
   * @param dropout drop out rate
   * @retval Tensor& reference of drop out mask
   */
  Tensor dropout_mask(float dropout) const;

  /**
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate) inplace
   * @param dropout drop out rate
   */
  void dropout_mask(float dropout);

  /**
   * @brief Calculate filter mask
   * @param mask_len length of each mask along the last axis
   * @param invert invert the mask
   */
  void filter_mask(const Tensor &mask_len, bool reverse = false);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param zoneout zone out rate
   * @retval Tensor zone out mask for opposite tensor
   */
  Tensor zoneout_mask(float zoneout);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param opposite opposite zone out mask
   * @param zoneout zone out rate
   */
  void zoneout_mask(Tensor &opposite, float zoneout);

  /**
   * @brief split tensor along axis.
   *
   * @param num_size num_size
   * @param axis axis
   * @return Tensor splitted tensor
   */
  std::vector<Tensor> split(unsigned num_size, int axis = 0);

  /**
   * @brief split tensor along axis.
   *
   * @param sizes sizes
   * @param axis axis
   * @return Tensor splitted tensor
   * @note if the given array sizes is just a 1 unsigned int value, assumes that
   * it divide tensor by given size evenly
   */
  std::vector<Tensor> split(std::vector<size_t> sizes, int axis = 0);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @param output output tensor to store the result
   * @return Tensor concatenated tensor
   *
   * @note  This function should not be used directly. Please use cat() instead.
   */
  Tensor concat(const std::vector<Tensor> &tensors, int axis, Tensor &output);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @return Tensor concatenated tensor
   */
  static Tensor cat(const std::vector<Tensor> &tensors, int axis = 0);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @param output output tensor to store the result
   * @return Tensor concatenated tensor
   */
  static Tensor cat(const std::vector<Tensor> &tensors, int axis,
                    Tensor &output);

  /**
   * @brief     Print element
   * @param[in] out out stream
   */
  void print(std::ostream &out) const;

  /**
   * @brief     put data of Tensor
   * @note      It is only effective when fsu is used
   */
  void putData() const;

  /**
   * @brief Set the memory buffer for the tensor
   *
   * @param buf the memory buffer
   * @param init intialize the buffer
   */
  void setData(const std::shared_ptr<MemoryData> buf, size_t off = 0,
               bool init = false);

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  const std::shared_ptr<MemoryData> getMemoryData() const;

  /**
   * @brief     return offset
   */
  size_t getOffset() const;

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   * @note support copying data from multiple data type
   */
  void copy(const Tensor &from, ComputeOps *ops = nullptr);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   * @note      support copying data from multiple data type
   */
  void copyData(const Tensor &from, ComputeOps *ops = nullptr);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   * @note      only support copying data from tensor with the same data type
   */
  void copy_with_stride(const Tensor &from);

  /**
   * @brief Get slice of the tensor, sliced by batch
   * @param[in] offset offset in batch to start the slice
   * @param[in] size size of the slice
   * @retval slice of this tensor
   * @note This function provides a slice of this tensor, and does not create a
   * copy
   */
  Tensor getBatchSlice(size_t offset, unsigned int size) const;

  /**
   * @brief Extract sub-tensor containing specified batch indices
   *
   * @param indices List of batch indices to extract (0-based) Duplicates are
   * allowed and will result in the same batch data being copied multiple times.
   * @return Tensor New tensor containing only specified batches  (copied
   * tensor!)
   *
   * @details
   * This function creates a new tensor containing copies of data from
   * specified batch indices of the original tensor. The operation:
   * - Requires the original tensor to be contiguous in memory
   * - Preserves channel/height/width dimensions
   * - Maintains data ordering within each batch
   * - Uses memcpy for efficient memory operations
   *
   * @note Duplicate indices: If the same index appears multiple times, the
   * corresponding batch data will be copied to each position in the output
   * tensor. Example: indices {0, 1, 1} creates output with 3 batches where
   * positions 1 and 2 contain identical copies  of input batch 1.
   *
   * @note
   * - Time complexity: O(k*C*H*W) where k = num_indices
   * - Memory complexity: O(k*C*H*W)
   * - Thread-safe when using different indices in parallel
   *
   * @throw std::runtime_error If:
   * - Tensor is not contiguous
   * - Any index is out of bounds
   */
  Tensor getBatchSlice(const std::vector<unsigned int> &indices) const;

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @retval    Copied version of this
   */
  Tensor clone() const;

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @param[in] type output tensor data type
   * @retval    Copied version of this
   */
  Tensor clone(ml::train::TensorDim::DataType type) const;

  /**
   * @brief     Read the Tensor For FSU
   *
   */
  void readFSU();

  /**
   * @brief     Save the Tensor into file
   * @param[in] file output file stream
   */
  void save(std::ostream &file);

  /**
   * @brief     Read the Tensor from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file, size_t start_offset = 0,
            bool read_from_offset = false, int file_fd = -1);

  /**
   * @brief     ReadSource
   * @param[in] ReadSource input file source
   */
  void read(ReadSource src, size_t start_offset = 0,
            bool read_from_offset = false);

  /**
   * @brief     return argument index which value is max by batch
   * @retval    unsigned int argument indices
   */
  std::vector<unsigned int> argmax() const;

  /**
   * @brief     return argument index which value is min by batch
   * @retval    unsigned int argument indices
   */
  std::vector<unsigned int> argmin() const;

  /**
   * @brief Find top-K maximum values along the width dimension and return
   * results as tensors
   *
   * @details This function computes the top-K maximum values and their
   * corresponding indices along the **width** dimension for each batch,
   * channel, and height slice. The operation preserves the original tensor
   * format (NCHW/NHWC) while reducing the width dimension to size K. The
   * indices are returned as a separate tensor of type `UINT32`.
   *
   * @param[in] k Number of largest elements to select (1 <= k <= width_size)
   *
   * @return std::pair<Tensor, Tensor> containing:
   *         - First: Output tensor of shape [batch][channel][height][k] (NCHW)
   * or [batch][height][k][channel] (NHWC) with top-K values
   *         - Second: Indices tensor of shape [batch][channel][height][k]
   * (NCHW) or [batch][height][k][channel] (NHWC) with original width positions
   *
   * @throw std::invalid_argument If:
   *         - k is 0 or exceeds width dimension size
   *         - Called on non-floating point tensor (UINT8/UINT16/etc)
   *
   * @note
   * - Indices represent positions in the **original width dimension**
   * - Sorting is done in descending order
   * - Preserves tensor format (NCHW/NHWC) of the original tensor
   */
  std::pair<Tensor, Tensor> topK(unsigned int k) const;

  /**
   * @brief     return max of the absolute values of the tensor
   * @retval    maximum absolute value
   */
  float max_abs(ComputeOps *ops = nullptr) const;

  /**
   * @brief  return maximum value
   * @retval Maximum value of the tensor data
   */
  float maxValue() const;

  /**
   * @brief  return minimum value
   * @retval Minimum value of the tensor data
   */
  float minValue() const;

  /**
   * @brief  Transpose Tensor
   * @param  direction to transpose ex) 0:2:1
   * @return Tensor
   */
  Tensor transpose(const std::string &direction) const;

  /**
   * @brief      Transpose Tensor
   * @param      direction to transpose ex) 0:2:1
   * @param[out] Tensor to save to, dimension is always reshaped.
   * @retval     Tensor& reference to the out
   */
  Tensor &transpose(const std::string &direction, Tensor &out,
                    ComputeOps *ops = nullptr) const;

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief fill tensor data with current value,
   * if dimension is not exactly same, it is a hard error in this function
   * so, only stride is overriden to @a this
   *
   * @param from Tensor to fill the data from
   * @param allocate if unallocated, allocate with from.getDim()
   * @throws std::invalid_argument if dimension and stride does not match
   */
  void fill(const Tensor &from, bool allocate = false);

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const;

  /**
   * @brief     return Tensor Type
   */
  TensorDim::TensorType getTensorType() const;

  /**
   * @brief Get initializer for the tensor
   *
   * @return initializer of the tensor
   */
  Initializer getInitializer() const;

  /**
   * @brief Get format for the tensor
   * @return format of the tensor
   */
  TensorDim::Format getFormat() const;

  /**
   * @brief Get data type for the tensor
   *
   * @return data type of the tensor
   */
  Tdatatype getDataType() const;

  /**
   * @brief     update batch size for this tensor
   * @param     batch size
   * @note      The batchsize of src_tensor need not be related with this
   * tensor's batch size
   *
   * @note      The memory for this tensor will re-allocated/re-assigned if the
   * updated batch size is different than the current batch size.
   *
   * @note      If this tensor is/was the src_tensor for some other, then
   * reduction in batch size can make the dependent tensors allocate fail due to
   * memory smaller. Caller must handle this in their own end.
   *
   * @note      If this tensor is re-allocated, then the memory might not be
   * immediately freed as the tensor already depending on this tensor also
   * share the same memory. So, the peak memory consumption in worst case can
   * reach the total memory requirements of a model with old batchsize and the
   * new batch size. It is recommended to first deallocate all the tensors,
   * updateBatch and then allocate again to avoid such issues.
   */
  void updateBatch(unsigned int batch);

  /**
   * @brief     update the dimension for this tensor
   * @param     dimension dimension to be updated
   * @note      if this tensor is allocated this will throw an error.
   * @note      we assume that the caller checks if the tensor is not allocated
   */
  void updateDimension(TensorDim dimension);

  /**
   * @brief     return whether tensor is contiguous or not.
   * @retval    bool contiguous
   */
  const bool getContiguous() const noexcept;

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept;

  /**
   * @brief     Check if two given axes are contiguous
   * @param[in] np1 first axis
   * @param[in] np2 second axis to compare with first axis
   * @retval    bool continuous
   */
  bool checkContinuous(unsigned int np1, unsigned int np2) const;

  /**
   * @brief     set FileOffset to Tensor
   * @param     off FileOffset
   */
  void setFileOffset(size_t file_offset);

  /**
   * @brief     get FileOffset of Tensor
   * @return    size_t fileOffset
   */
  size_t getFileOffset() const;

  /**
   * @brief     Set name of the tensor
   * @param[in] name_ tensor name
   */
  void setName(const std::string &name_);

  /**
   * @brief     Get name of the tensor
   * @retval    string name
   */
  const std::string &getName() const;

  /**
   * @brief Get linear index given the n-d index
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept;
  /**
   * @brief     Get size of current tensor
   * @retval    unsigned int size of the current tensor
   */
  size_t size() const;

  /**
   * @brief     Get if the tensor is empty
   * @retval    true if the tensor is empty
   */
  bool empty() const;

  /**
   * @brief     Get size of the data in bytes
   * @retval    size_t Size in bytes
   */
  size_t bytes() const;

  /**
   * @brief     Get a total size of the memory data in bytes
   * @retval    size_t Size in bytes
   * @note      This is the total size of the memory data, including the scale
   * factors and the zero points. For float type, this will return the same as
   * bytes()
   */
  size_t getMemoryBytes() const;

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t batch() const;

  /**
   * @brief     return Tensor channel size
   * @retval    channel size
   */
  size_t channel() const;

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  size_t height() const;

  /**
   * @brief     return Tensor width size
   * @retval    width size
   */
  size_t width() const;

  /**
   * @brief     return Tensor scale factor size if exists
   * @retval    scale factor size
   */
  size_t scale_size() const;

  /**
   * @brief     return Tensor quantization scheme
   * @retval    Qscheme qscheme
   */
  QScheme q_scheme() const;

  /**
   * @brief     return the quantization group size for grouped per-channel
   *            quantized tensors (e.g. Int4QTensor PER_CHANNEL_AFFINE).
   *            Returns 0 for non-quantized tensors. For Int4QTensor, a
   *            value of 0 also signals pure per-channel (qsi4cxp: one
   *            scale per output channel); any non-zero value means
   *            "group_size elements share one scale" within each channel.
   * @retval    group size in elements, 0 if not applicable
   */
  size_t group_size() const;

  /**
   * @brief Merge the given two axis for tensor at second axis inplace
   *
   * @param axis1 first axis to merge
   * @param axis2 second axis to merge
   */
  void mergeAxis(unsigned int axis1, unsigned int axis2);

  /**
   * @brief Update destination tensor to share memory with source tensor
   *
   * @param src src tensor containing the memory
   * @param dest destination tensor which will share the memory
   * @param offset offset to be used from the start of the data in bytes
   * @note The new tensor will share the same data as the current tensor but
   * can have different size.
   * @note New size added with offset must be less than the size of the original
   * tensor.
   */
  void createSharedDataTensor(const Tensor &src, Tensor &dest,
                              size_t offset) const;

  /**
   * @brief Get new tensor which shares memory with current tensor but different
   * shape
   *
   * @param dim new dimension to be set for this tensor
   * @param offset offset to be used from the start of the data in elements
   * @note The new tensor will share the same data as the current tensor but
   * can have different size.
   * @note New size added with offset must be less than the size of the original
   * tensor.
   */
  Tensor getSharedDataTensor(const TensorDim dim_, size_t offset,
                             bool reset_stride = true,
                             const std::string &name_ = "") const;

  /**
   * @brief    Swaps Tensor lhs and rhs
   * @param[in] lhs Tensor to be swapped
   * @param[in] rhs Tensor to be swapped
   */
  friend void swap(Tensor &lhs, Tensor &rhs) noexcept {
    std::swap(lhs.itensor_, rhs.itensor_);
  }

  /**
   * @brief      check if there is NaN or Inf element
   * @param[out] bool false if there is NaN or Inf else false
   */
  bool isValid(ComputeOps *ops = nullptr) const { return itensor_->isValid(ops); };

  /**
   * @brief check if tensor is virtual
   * @param[out] bool false if tensor is not virtual else true
   */
  bool isVirtual() const { return is_virtual; }

  /**
   * @brief activate virtual tensor
   * @note if the tensor is virtual, this method activates virtual tensor, which
   * means allocate the tensor memory and read the corresponding value from the
   * file descriptor
   * @todo it is not supported on Windows yet
   */
  void activate();

  /**
   * @brief deactivate virtual tensor
   * @note if the tensor is virtual and already activated, the tensor is
   * deallocated.
   */
  void deactivate();

  static constexpr float epsilon = 1e-5f;

private:
  std::unique_ptr<TensorBase> itensor_;

  /**
   * @brief properties for virtual tensor
   * @note This should be removed by defining VirutalTensor class
   * */
  bool is_virtual = false;    /** flag to check virtual */
  size_t read_offset;         /** save read_offset info for virtual */
  int fd = -1;                /** save fd info for virtual */
  void *mapped_ptr = nullptr; /** save mmap buf pointer for virtual */

  /**
   * @brief Set tensor variables
   *
   * @param[in] d TensorDim
   * @param[in] buf buffer
   * @param[in] offset offset to be used
   */
  void setTensorVar(TensorDim d, void *buf, size_t offset);

  /**
   * @brief Calculate the output tensor dimension of the concatenating a list of
   * tensors as an input.
   *
   * @param[in] tensors tensors to be concatenated to the first tensor
   * @param[in] axis axis
   */
  static TensorDim calculateConcatOutputDim(const std::vector<Tensor> &tensors,
                                            int axis);
};

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, Tensor const &input);

typedef std::shared_ptr<Tensor> sharedTensor;

typedef std::shared_ptr<const Tensor> sharedConstTensor;

typedef std::vector<sharedConstTensor> sharedConstTensors;

typedef std::vector<sharedTensor> sharedTensors;

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_H__ */
