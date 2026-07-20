// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   tensor_dim.h
 * @date   22 May 2020
 * @brief  This is Tensor Dimension Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_DIM_H__
#define __TENSOR_DIM_H__
#ifdef __cplusplus

#include <array>
#include <iosfwd>

#include <bitset>
#include <vector>

#ifdef ENABLE_FP16
#ifdef USE__FP16
#define _FP16 __fp16
#else
#define _FP16 _Float16
#endif
#endif

/**
 * @brief Namespace for machine learning APIs
 */
namespace ml {
/**
 * @brief Namespace for training APIs
 */
namespace train {

/**
 * @brief Tensor Dimension. This class is used to save dimension information
 *
 */
class TensorDim {
public:
  static constexpr const size_t MAXDIM = 4;

  /**
   * @brief Tensor Format. Channel Last or Channel First
   *
   */
  enum class Format { NCHW, NHWC };

  /**
   * @brief Tensor Data Type.
   * Currently support QINT4, QINT8, QINT16, BCQ, Q4_K, UINT8, UINT16, UINT32,
   * FP16 & FP32
   */
  enum class DataType {
    QINT4,  /** quantized int 4*/
    QINT8,  /** quantized int 8*/
    QINT16, /** quantized int 16*/
    BCQ,    /** binary-code-based quantized*/
    Q4_K,   /** Q4_K quantized*/
    Q6_K,   /** q6 k quantized */
    Q4_0,   /** Q4_0 k quantized */
    Q8_0,  /** Q8_0 k quantized (8-bit signed + fp16 scale per 32-element block)
            */
    QS4CX, /** QS4CX quantized */
    UINT4, /** quantized unsigned int 4*/
    UINT8, /** unsigned int 8 bit */
    UINT16, /** unsigned int 16 bit */
    UINT32, /** unsigned int 32 bit */
    FP16,   /** half precision */
    FP32,   /** single precision */
    NONE,   /** not specified */
  };

  /**
   * @brief Tensor Data Storage Order. Row-major or Column-major
   *
   */
  enum class StorageOrder { ROW_MAJOR, COL_MAJOR };

  /**
   * @brief Tensor Type which context to hold the Format & DataType
   *
   */
  struct TensorType {
    /**
     * @brief     Tensor Format : Default is NCHW
     */
    Format format;

    /**
     * @brief     Tensor Data Type : Default is FP32
     */
    DataType data_type;

    /**
     * @brief     Data Storage Order : Default is Row-major
     */
    StorageOrder storage_order;

    /**
     * @brief     Default creator of Tensor Type
     */
    TensorType() :
      format(Format::NCHW),
      data_type(DataType::FP32),
      storage_order(StorageOrder::ROW_MAJOR){};

    /**
     * @brief     Default creator of Tensor Type with Format & DataType
     */
    TensorType(Format fm, DataType d_type,
               StorageOrder order = StorageOrder::ROW_MAJOR) :
      format(fm), data_type(d_type), storage_order(order){};
  };

  /**
   * @brief Get the Num Dim object
   *
   * @return unsigned int fixed value of MAXDIM
   */
  static unsigned int getNumDim();

  /**
   * @brief     Creator of TensorDim with Format & DataType
   *
   * @param fm format NCHW | HNWC
   * @param d_type DataType QINT4 | QINT8 | QINT16 | BCQ | UINT8 | UINT16 |
   * UINT32 | FP16 | FP32
   * @param eff_dim_flag_ effective dimension flag (1 means it's effective)
   * @param dyn_dim_flag_ dynamic dimension flag (1 means it's unspecified)
   */
  TensorDim(TensorDim::Format fm, TensorDim::DataType d_type,
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param t_type_ tensor type
   * @param eff_dim_flag_ effective dimension flag (1 means it's effective)
   * @param dyn_dim_flag_ dynamic dimension flag (1 means it's unspecified)
   */
  explicit TensorDim(TensorType t_type_ = TensorType(),
                     const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
                     const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param dims std::initialize_list
   * @param t_type_ tensor type
   *
   * formats of {w}, {h, w}, {c, h, w}, {b, c, h, w} for the NCHW & NHWC are
   * accepted
   */
  TensorDim(std::initializer_list<size_t> dims,
            TensorType t_type_ = TensorType());

  /**
   * @brief Construct a new Tensor Dim object without batch dimension
   *
   * @param shapes shapes without batch dimension
   * @param t_type_ tensor type
   */
  TensorDim(const std::array<size_t, 3> &shapes,
            TensorType t_type_ = TensorType());

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param b batch
   * @param c channel
   * @param h height
   * @param w width
   * @param t_type format NCHW | HNWC , dataType FP32 | FP16
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(size_t b, size_t c, size_t h, size_t w,
            TensorType t_type_ = TensorType(),
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param c channel
   * @param h height
   * @param w width
   * @param t_type format NCHW | HNWC , dataType FP32 | FP16
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(size_t c, size_t h, size_t w, TensorType t_type_ = TensorType(),
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param h height
   * @param w width
   * @param t_type format NCHW | HNWC , dataType FP32 | FP16
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(size_t h, size_t w, TensorType t_type_ = TensorType(),
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param w width
   * @param t_type format NCHW | HNWC , dataType FP32 | FP16
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(size_t w, TensorType t_type_ = TensorType(),
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param b batch
   * @param c channel
   * @param h height
   * @param w width
   * @param fm format NCHW | HNWC
   * @param d_type DataType QINT4 | QINT8 | QINT16 | BCQ | UINT8 | UINT16 |
   * UINT32 | FP16 | FP32
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(size_t d0, size_t d1, size_t d2, size_t d3, TensorDim::Format fm,
            TensorDim::DataType d_type,
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000);

  /**
   * @brief Copy construct a new tensor dim
   *
   * @param rhs tensor dim to copy from
   */
  TensorDim(const TensorDim &rhs) = default;

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param shape shape of format
   * @param t_type_ Tensor Type
   */
  TensorDim(const std::string &shape, TensorType t_type_ = TensorType());

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param shape shape of format
   * @param fm format NCHW | HNWC
   * @param d_type DataType QINT4 | QINT8 | QINT16 | BCQ | UINT8 | UINT16 |
   * UINT32 | FP16 | FP32
   * @param order data storage order ROW_MAJOR | COL_MAJOR
   */
  TensorDim(const std::string &shape, TensorDim::Format fm,
            TensorDim::DataType d_type = TensorDim::DataType::FP32,
            TensorDim::StorageOrder order = TensorDim::StorageOrder::ROW_MAJOR);

  /**
   * @brief Destroy the Tensor Dim object
   *
   */
  ~TensorDim() = default;

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  TensorDim(TensorDim &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Optimizer to be moved.
   */
  TensorDim &operator=(TensorDim &&rhs) noexcept;

  /**
   * @brief  get data type size
   */
  unsigned int getDataTypeSize() const;

  /**
   * @brief Set the Dim Flag to retrieve effective dimension
   * @note eg) if dimension 4:1:10:1 should be squeezed to 4:10,
   *       set this to 0b1010, rightmost is width
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setEffDimFlag(const std::bitset<MAXDIM> &dim_flag_);

  /**
   * @brief Set the dynamic Dim Flag to retrieve dynamic dimension (that can
   * change during running)
   * @note eg) if dimension 4:1:10:1 should be squeezed to dynamic to batch,
   *       set this to 0b1000, rightmost is width
   * @note when setting dynamic dimension, the calculation must remain
   * independent of the dynamic dimension. Please check this :)
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setDynDimFlag(const std::bitset<MAXDIM> &dim_flag_);

  /**
   * @brief Get the Dim Flag to retrieve effective dimension
   * @note eg) if dimension 4:1:10:1 should be squeezed to 4:10,
   *       set this to 0b1010, rightmost is width
   *
   * @return dim_flag_ dimension bit to calculate, rightmost is width
   */
  const std::bitset<MAXDIM> &getEffDimFlag() const;

  /**
   * @brief Get the dynamic Dim Flag to retrieve dynamic dimension (that can
   * change during running)
   * @note eg) if dimension 4:1:10:1 should be squeezed to dynamic to batch,
   *       set this to 0b1000, rightmost is width
   * @note when setting dynamic dimension, the calculation must remain
   * independent of the dynamic dimension. Please check this :)
   *
   * @return dim_flag_ dimension bit to calculate, rightmost is width
   */
  const std::bitset<MAXDIM> &getDynDimFlag() const;

  /**
   * @brief  swap variable of Conv2D Layer
   * @parma[out] lhs Optimizer
   * @parma[in] rhs Optimizer
   */
  friend void swap(TensorDim &lhs, TensorDim &rhs) noexcept;

  /**
   * @brief get batch (axis 0)
   *
   * @return unsigned int batch size
   */
  size_t batch() const;

  /**
   * @brief get channel (axis 1)
   *
   * @return size_t channel size
   */
  size_t channel() const;

  /**
   * @brief get height (axis 2)
   *
   * @return size_t height size
   */
  size_t height() const;

  /**
   * @brief get width (axis 3)
   *
   * @return size_t width size
   */
  size_t width() const;

  /**
   * @brief Get the Data Len object
   *
   * @return size_t get length of the data
   */
  size_t getDataLen() const;

  /**
   * @brief Get the Feature Len object
   *
   * @return size_t get feature length
   */
  size_t getFeatureLen() const;

  /**
   * @brief set batch (axis 0)
   *
   * @param b batch to set
   */
  void batch(size_t b);

  /**
   * @brief set channel (axis 1)
   *
   * @param c channel to set
   */
  void channel(size_t c);

  /**
   * @brief set height (axis 2)
   *
   * @param h height to set
   */
  void height(size_t h);

  /**
   * @brief set width (axis 3)
   *
   * @param w width to set
   */
  void width(size_t w);

  /**
   * @brief Get the Dim object
   *
   * @return const size_t* array of size[MAXDIM]
   */
  const size_t *getDim() const;

  /**
   * @brief calculate tranposed dimension
   * @note In this function, batch direction is not considered, so channel is 0
   * @todo make batch 0
   *
   * @param direction  direction to transpose
   * @return TensorDim calculated dimension
   */
  TensorDim transpose(const std::string &direction) const;

  /**
   * @brief calculate trasposed dimension
   * @note In this function, batch direction is considered 0
   *
   * @param axes axes to be transposed
   * @return TensorDim calculated dimension
   */
  TensorDim transpose(const std::array<size_t, MAXDIM> &axes) const;

  /**
   * @brief Get the Tensor dimension for an axis
   *
   * @param idx axis to get
   * @return const size_t dimension of the given axis
   */
  const size_t getTensorDim(unsigned int idx) const;

  /**
   * @brief Set the Tensor Dim object
   *
   * @param idx axis to set
   * @param value value to set
   */
  void setTensorDim(unsigned int idx, size_t value);

  /**
   * @brief Set the Tensor Dim object
   *
   * @param input_shape input_shape
   * @param fm NCHW | NHWC
   * @return int ML_ERROR_NONE if successs
   */
  int setTensorDim(const std::string &input_shape,
                   TensorType t_type_ = TensorType());

  /**
   * @brief copy assign a dimension
   *
   * @param rhs other side to copy assign
   * @return TensorDim& tensor dimension
   */
  TensorDim &operator=(const TensorDim &rhs);

  /**
   * @brief check if tensor dims are equal
   *
   * @param rhs other side to compare
   * @retval true equal
   * @retval false not equal
   */
  bool operator==(const TensorDim &rhs) const;

  /**
   * @brief check if tensor dims are not equal
   *
   * @param rhs other side to compare
   * @retval true not equal
   * @retval false equal
   */
  bool operator!=(const TensorDim &rhs) const;

  /**
   * @brief check if given tensor dimension is empty
   *
   * @retval true empty
   * @retval false not empty
   */
  bool isEmpty() const;

  /**
   * @brief get index rank (dimension of 1 is considered not valid here)
   *
   * @return unsigned int calculated index
   */
  unsigned int rank() const;

  /**
   * @brief operator[] to get index from tensor_dim
   *
   * @param index index
   * @return unsigned int& returned index reference
   */
  size_t &operator[](const unsigned int index);

  /**
   * @brief operator[] to get index from tensor_dim
   *
   * @param index index
   * @return const size_t& returned index reference
   */
  const size_t &operator[](const unsigned int index) const;

  /**
   * @brief Calculate standard strides
   *
   * @return std::array <unsigned int, MAXDIM>
   */
  std::array<size_t, MAXDIM> computeStrides() const;

  /**
   * @brief reverse the dimensions inplace
   */
  void reverse();

  /**
   * @brief Get the Effective Dimension of the current
   * @note dynamic dimension is returned as -1
   *
   * @param dynamic if dimension has to be considering dynamic set this to ture
   * @return std::vector<int> integer vector
   */
  std::vector<int> getEffectiveDimension(bool dynamic = false) const;

  /**
   * @brief check if tensor is dynamic
   *
   * @retval true any of dyn_dim_flag is set
   * @retval false none of dyn_dim_flag is set
   */
  bool is_dynamic() const;

  /**
   * @brief getFormat
   *
   */
  TensorDim::Format getFormat() const { return t_type.format; };

  /**
   * @brief getType
   *
   */
  TensorDim::DataType getDataType() const { return t_type.data_type; };

  /**
   * @brief getStorageOrder
   *
   */
  TensorDim::StorageOrder getStorageOrder() const {
    return t_type.storage_order;
  };

  /**
   * @brief setFormat
   *
   */
  void setFormat(TensorDim::Format fm) { t_type.format = fm; };

  /**
   * @brief setDataType
   *
   */
  void setDataType(TensorDim::DataType ty) { t_type.data_type = ty; };

  /**
   * @brief setDataType
   *
   */
  void setStorageOrder(TensorDim::StorageOrder storage_order_) {
    t_type.storage_order = storage_order_;
  };

  /**
   * @brief getFormat
   *
   */
  TensorType getTensorType() const { return t_type; };

  /**
   * @brief setTensorType
   *
   */
  void setTensorType(TensorType tt) { t_type = tt; };

private:
  /**
   * @brief reset length
   *
   */
  void resetLen();

  TensorType t_type;

  std::bitset<MAXDIM> eff_dim_flag; /**< dimension bit flag to define effective
          dimension size */

  std::bitset<MAXDIM> dyn_dim_flag; /**< dimension bit flag to define
dynamic dimension size */

  size_t dim[MAXDIM]; /**< underlying dimension type */
  size_t len;         /**< number of elements */
  size_t feature_len; /**< number of feature elements */
};

/**
 * @brief operator<< to print TensorDim
 *
 * @param out ostream
 * @param d dimension to print
 * @return std::ostream& ostream
 */
std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace train */
} /* namespace ml */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
