// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_base.cpp
 * @date	04 December 2023
 * @brief	This is Tensor base class
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <tensor.h>
#include <tensor_base.h>

namespace nntrainer {

ComputeOps *TensorBase::getOps() const {
  if (ct_data_) {
    if (auto *o = ct_data_->getComputeOps())
      return o;
  }
  return nntrainer::getComputeOps();
}

TensorBase::TensorBase(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name_) :
  TensorBase(name_, d.getFormat()) {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    initializer = init;
  }
}

bool TensorBase::operator==(const TensorBase &rhs) const {
  if (this->dim != rhs.dim)
    return false;

  if (size() != rhs.size())
    return false;

  if (contiguous != rhs.contiguous)
    return false;

  if (strides != rhs.strides)
    return false;

  return true;
}

void TensorBase::setTensorVar(TensorDim d, void *buf, size_t offset) {
  dim = d;
  strides = d.computeStrides();
  /// Tensor does not own the memory
  data = std::make_shared<MemoryData>(buf);
  this->offset = offset;
}

void TensorBase::save(std::ostream &file) {
  std::streamsize sz = static_cast<std::streamsize>(bytes());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz, "[Tensor::save] operation failed");
  putData();
}

void TensorBase::read(std::ifstream &file, size_t start_offset,
                      bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max()) {
    start_offset = file_offset;
  }
  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, (char *)getData(), sz, "[Tensor::read] operation failed",
              start_offset, read_from_offset);
  putData();
}

void TensorBase::read(ReadSource src, size_t start_offset,
                      bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max()) {
    start_offset = file_offset;
  }
  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(src, (char *)getData(), sz, "[Tensor::read] operation failed",
              start_offset, read_from_offset);
  putData();
}

void TensorBase::readFSU() {}

void TensorBase::putData() const {
  if (!data)
    return;

  data->invalidate();
}

void TensorBase::setMemoryData(const std::shared_ptr<MemoryData> buf,
                               size_t off) {
  if (buf) {
    data = buf;
    offset = off;
  } else {
    data = nullptr;
    offset = 0;
  }
}

const std::shared_ptr<MemoryData> TensorBase::getMemoryData() const {
  return data;
}

size_t TensorBase::getOffset() const { return offset; }

size_t TensorBase::getFileOffset() const { return file_offset; }

void TensorBase::setFileOffset(size_t off) { file_offset = off; }

void TensorBase::reshape(const TensorDim &d) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot reshape.";

  NNTR_THROW_IF(d.getDataLen() != dim.getDataLen(), std::invalid_argument)
    << "[Tensor]: reshape cannot change the buffer size, trying reshaping "
       "\nfrom "
    << getDim() << " to " << d;

  dim.batch(d.batch());
  dim.channel(d.channel());
  dim.height(d.height());
  dim.width(d.width());

  strides = d.computeStrides();
}

void TensorBase::updateBatch(unsigned int batch) {
  if (dim.batch() == batch) {
    return;
  }

  if (isAllocated())
    throw std::invalid_argument("Cannot update batch for an allocated tensor");
  dim.batch(batch);
}

void TensorBase::updateDimension(TensorDim dimension) {
  if (dim == dimension) {
    return;
  }

  if (isAllocated())
    throw std::invalid_argument(
      "Cannot update tensor dimension for an allocated tensor");

  dim = dimension;
}

size_t TensorBase::getIndex(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) const noexcept {
  if (getFormat() == Tformat::NCHW) {
    return (b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]);
  } else {
    return (b * strides[0] + h * strides[1] + w * strides[2] + c * strides[3]);
  }
}

void TensorBase::mergeAxis(unsigned int axis1, unsigned int axis2) {
  dim.setTensorDim(axis2, dim.getTensorDim(axis1) * dim.getTensorDim(axis2));
  dim.setTensorDim(axis1, 1);
}

void TensorBase::allocateSrcTensor() {
  if (src_tensor) {
    data = src_tensor->tensor()->data;
    offset = src_tensor->tensor()->offset + src_tensor->offset();
  }
}

void TensorBase::createSharedDataTensor(const TensorBase *src, TensorBase *dest,
                                        size_t offset) const {
  /**
   * - If src already has data allocated, then directly make dest tensor based
   * on the src tensor.
   * - If src->data does not exist (meaning tensor does not memory allocated),
   * and src->src_tensor does not exist (meaning the src tensor does not depened
   * on another tensor), then create a SrcSharedTensor around the src.
   * - If src->src_tensor exists, then use the src->src_tensor to create the
   *  required SrcSharedTensor to avoid recursive dependency.
   *
   * @note src->data and src->src_tensor CAN co-exist. src->src_tensor is stored
   * if the batch size of src is updated and needs reallocation.
   */
  dest->data = nullptr;
  if (src->data) {
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(src, offset);
    dest->allocate();
  } else if (!src->src_tensor)
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(src, offset);
  else
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(
      src->src_tensor->tensor(), offset + src->src_tensor->offset());
}

void TensorBase::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                     bool reset_stride,
                                     const std::string &name_,
                                     TensorBase *ret) {
  if (dim_.getFormat() != ret->dim.getFormat())
    throw std::invalid_argument("Tensor format does not match");

  ret->dim = dim_;
  if (!name_.empty())
    ret->name = name_;

  if (dim_.getDataLen() + offset > dim.getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");

  if (reset_stride)
    ret->strides = ret->dim.computeStrides();

  TensorDim new_match_dim = dim_;
  new_match_dim.batch(dim.batch());
  if (new_match_dim != dim && !reset_stride)
    ret->contiguous = false;

  /**
   * In this case, its the caller's responsibility to ensure that allocate() is
   * called for the output tensor before operating on the output tensor.
   */
  createSharedDataTensor(this, ret, offset);
}

TensorBase::BroadcastInfo
TensorBase::computeBroadcastInfo(const Tensor &m) const {
  if (m.size() > this->size())
    throw exception::not_supported("broadcasting *this is not supported");

  const TensorDim m_dim = m.getDim();

  BroadcastInfo e;
  e.tensor_type = getTensorType();

  unsigned int continuity[4] = {0, 1, 2, 3};
  if (getFormat() == Tformat::NHWC) {
    continuity[1] = 2;
    continuity[2] = 3;
    continuity[3] = 1;
  }

  /// checking if given Tensor's can be broadcasted
  for (unsigned int i = 0; i < TensorDim::MAXDIM; ++i) {
    if (dim.getTensorDim(continuity[i]) == m_dim.getTensorDim(continuity[i])) {
      e.strides[i] = m.getStrides()[i];
      continue;
    }

    /// If given dimension is 1, it could be reused, the stride remaining 0
    /// Need to check if dim[i] == 1 && m_dim[i] == 1 first though
    /// If so, strides should not change
    if (m_dim.getTensorDim(continuity[i]) == 1) {
      continue;
    }

    std::stringstream ss;
    ss << "[computeBroadcastInfo] broadcasting only allowed for "
          "dimension value of 1 \n"
       << "this: " << dim << "target: " << m_dim;
    throw std::invalid_argument(ss.str().c_str());
  }

  /// calculate inner loop size
  e.buffer_size = 1;
  e.buffer_axis = -1;
  e.strides[3] = m.getStrides()[3];

  /// initiate buffer info with matching dimension strategy
  for (int axis = 3; axis >= 0; --axis) {
    if (dim.getTensorDim(continuity[axis]) !=
        m_dim.getTensorDim(continuity[axis])) {
      e.buffer_axis = axis;
      break;
    }

    e.buffer_size *= dim.getTensorDim(continuity[axis]);
  }

  /// check strategy that uses consecutive ones
  if (m_dim.getTensorDim(continuity[3]) == 1) {
    unsigned int inner_loop_size = 1;
    int axis;
    for (axis = 3; axis >= 0; --axis) {
      if (m_dim.getTensorDim(continuity[axis]) != 1) {
        break;
      }

      inner_loop_size *= dim.getTensorDim(continuity[axis]);
    }

    /// if consecutive-one strategy has bigger chunk size, replace the
    /// information
    if (inner_loop_size > e.buffer_size) {
      e.buffer_axis = axis;
      e.buffer_size = inner_loop_size;
      e.strides[3] = 0;
    }
  }

  return e;
}

void TensorBase::calculateFlattenDot(
  Tensor const &input, Tensor &output, bool trans, bool trans_in,
  unsigned int &first_three_flat, unsigned int &last_axis,
  unsigned int &input_first_three_flat, unsigned int &input_last_axis,
  unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &lda,
  unsigned int &ldb, unsigned int &ldc) const {

  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }

  if (getFormat() == Tformat::NHWC) {
    first_three_flat = batch() * height() * width();
    last_axis = channel();
    input_first_three_flat = input.batch() * input.height() * input.width();
    input_last_axis = input.channel();
  } else {
    first_three_flat = batch() * channel() * height();
    last_axis = width();
    input_first_three_flat = input.batch() * input.channel() * input.height();
    input_last_axis = input.width();
  }

  if (!trans && !trans_in) {
    if (last_axis != input_first_three_flat)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = input_first_three_flat; /** == last_axis */
    N = input_last_axis;
    M = first_three_flat;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(output, batch(), N, height(), width(),
                           getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(output, batch(), channel(), height(), N,
                           getTensorType());
    }

    // We are not set zero the output because of performance reason.
    // However, output is not initialized properly. There might include
    // garbage like nan. When we have to use this value as in C = alpha*A*B +
    // beta*C, then have to check garbage data of C is not effect or not.

  } else if (!trans && trans_in) {
    if (last_axis != input_last_axis)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = input_last_axis; /** == last_axis */
    N = input_first_three_flat;
    M = first_three_flat;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(output, batch(), N, height(), width(),
                           getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(output, batch(), channel(), height(), N,
                           getTensorType());
    }
  } else if (trans && !trans_in) {
    if (first_three_flat != input_first_three_flat)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = input_first_three_flat; /** == first_three_flat */
    N = input_last_axis;
    M = last_axis;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(output, 1, N, M, 1, getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(output, 1, 1, M, N, getTensorType());
    }
  } else {
    if (first_three_flat != input_last_axis)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = input_last_axis; /** == first_three_flat */
    N = input_first_three_flat;
    M = last_axis;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(output, 1, N, M, 1, getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(output, 1, 1, M, N, getTensorType());
    }
  }

  lda = last_axis;
  ldb = input_last_axis;
  ldc = (getFormat() == Tformat::NHWC) ? output.channel() : output.width();
}

/**
 * Please note that the following functions need to be implemented in a child
 * class to utilize tensor operations fully — operations such as addition,
 * division, multiplication, dot production, data averaging, and so on.
 */
void TensorBase::setRandNormal(float mean, float stddev) {
  throw std::invalid_argument(
    "Tensor::setRandNormal() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::setRandUniform(float min, float max) {
  throw std::invalid_argument(
    "Tensor::setRandUniform() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::setRandBernoulli(float probability) {
  throw std::invalid_argument("Tensor::setRandBernoulli() is currently not "
                              "supported in tensor data type " +
                              getStringDataType());
}

Tensor TensorBase::multiply_strided(Tensor const &m, Tensor &output,
                                    const float beta) const {
  throw std::invalid_argument("Tensor::multiply_strided() is currently not "
                              "supported in tensor data type " +
                              getStringDataType());
}

int TensorBase::multiply_i(float const &value) {
  throw std::invalid_argument(
    "Tensor::multiply_i() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::multiply(float const &value, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::multiply() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::multiply(Tensor const &m, Tensor &output,
                             const float beta) const {
  throw std::invalid_argument(
    "Tensor::multiply() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::divide(float const &value, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::divide() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::divide(Tensor const &m, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::divide() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::add_strided(Tensor const &input, Tensor &output,
                                const float beta) const {
  throw std::invalid_argument(
    "Tensor::add_strided() is currently not supported in tensor data type " +
    getStringDataType());
}

int TensorBase::add_i_partial(unsigned int len, unsigned int addr_idx,
                              Tensor &m, unsigned int incX, unsigned int incY,
                              const Tensor alphas, unsigned int alpha_idx) {
  throw std::invalid_argument(
    "Tensor::add_i_partial() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::add(float const &value, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::add() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::add(Tensor const &m, Tensor &output,
                        float const alpha) const {
  throw std::invalid_argument(
    "Tensor::add() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::subtract(float const &value, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::subtract() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::sum_by_batch(Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::sum_by_batch() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::sum(unsigned int axis, Tensor &output, float alpha,
                        float beta) const {
  throw std::invalid_argument(
    "Tensor::sum() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::abs(Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::abs() is currently not supported in tensor data type " +
    getStringDataType());
}

float TensorBase::l2norm() const {
  throw std::invalid_argument(
    "Tensor::l2norm() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::normalization_i(unsigned int dim, float p, float epsilon) {
  throw std::runtime_error(
    "Tensor::normalization_i() is not supported for this tensor type" +
    getStringDataType());
}

Tensor &TensorBase::pow(float exponent, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::pow() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::sqrt(Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::sqrt() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::erf(Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::erf() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::sin(Tensor &out, float alpha) {
  throw std::invalid_argument(
    "Tensor::sin() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::cos(Tensor &out, float alpha) {
  throw std::invalid_argument(
    "Tensor::cos() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::tan(Tensor &output, float alpha) {
  throw std::invalid_argument(
    "Tensor::tan() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::inv_sqrt(Tensor &out) {
  throw std::invalid_argument(
    "Tensor::inv_sqrt() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::dot(Tensor const &input, Tensor &output, bool trans,
                        bool trans_in, float beta) const {
  throw std::invalid_argument(
    "Tensor::dot() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::dot(std::vector<Tensor *> input, std::vector<Tensor *> output,
                     bool trans, bool trans_in, float beta) const {
  throw std::invalid_argument("Tensor::dot(std::vector<Tensor*>) is currently "
                              "not supported in tensor data type " +
                              getStringDataType());
}

void TensorBase::dropout_mask(float dropout) {
  throw std::invalid_argument(
    "Tensor::dropout_mask() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::filter_mask(const Tensor &mask_len, bool reverse) {
  throw std::invalid_argument(
    "Tensor::filter_mask() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::zoneout_mask(Tensor &opposite, float zoneout) {
  throw std::invalid_argument(
    "Tensor::zoneout_mask() is currently not supported in tensor data type " +
    getStringDataType());
}

std::vector<Tensor> TensorBase::split(std::vector<size_t> sizes, int axis) {
  throw std::invalid_argument(
    "Tensor::split() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor TensorBase::concat(const std::vector<Tensor> &tensors, int axis,
                          Tensor &output) {
  throw std::invalid_argument(
    "Tensor::concat() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::apply(std::function<float(float)> f, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::apply(std::function<float(float)> f, Tensor &output) is "
    "not supported in tensor data type " +
    getStringDataType());
}

#ifdef ENABLE_FP16
Tensor &TensorBase::apply(std::function<_FP16(_FP16)> f, Tensor &output) const {
  throw std::invalid_argument(
    "Tensor::apply(std::function<_FP16(_FP16)> f, Tensor &output) is "
    "not supported in tensor data type " +
    getStringDataType());
}
#endif

std::vector<unsigned int> TensorBase::argmax() const {
  throw std::invalid_argument(
    "Tensor::argmax() is currently not supported in tensor data type " +
    getStringDataType());
}

std::vector<unsigned int> TensorBase::argmin() const {
  throw std::invalid_argument(
    "Tensor::argmin() is currently not supported in tensor data type " +
    getStringDataType());
}

void TensorBase::topK(unsigned int k, void *output_data, uint32_t *indices) {
  throw std::invalid_argument(
    "Tensor::topK() is currently not supported in tensor data type " +
    getStringDataType());
}

Tensor &TensorBase::transpose(const std::string &direction, Tensor &out) const {
  throw std::invalid_argument(
    "Tensor::transpose() is currently not supported in tensor data type " +
    getStringDataType());
}

} // namespace nntrainer
