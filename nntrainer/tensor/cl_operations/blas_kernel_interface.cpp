// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.cpp
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <clblast_interface.h>

namespace nntrainer {
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans, bool trans_m) {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < input.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = input.getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    dotCl(this_b, m_b, result_b, trans, trans_m);
  }
}

Tensor dotCl(Tensor const &input, Tensor const &m, bool trans, bool trans_m) {
  Tensor output("", input.getFormat(), input.getDataType());
  dotCl(input, m, output, trans, trans_m);

  return output;
}

void dotCl(Tensor const &input, Tensor const &m, Tensor &result, bool trans,
           bool trans_m) {
  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(),
                           input.getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(), input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = m.getData();
    float *rdata = result.getData();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      // *rdata = dot_cl(data, mdata, K) + (*rdata);
      *rdata = dot_cl(K, data, mdata) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      gemv_cl(0, trans, dim1, dim2, 1.0f, data, lda, mdata, 0.0f, rdata, 1);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      gemv_cl(0, !trans_m, mdim1, mdim2, 1.0f, mdata, ldb, data, 0.0f, rdata,
              1);
    }
    /// case others: use gemm
    else {
      if (input.getFormat() == Tformat::NHWC) {
        sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
      } else {
        gemm_cl(0, trans, trans_m, M, N, K, 1.0f, data, (trans) ? M : K, mdata,
                (trans_m) ? K : N, 1.0f, rdata, N);
      }
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    const _FP16 *mdata = m.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();

    // out_svm: a result tensor in coarse-grain shared virtual memory has to be
    // host-mapped before the staging read-back, or the copy-out silently never
    // lands (see sgemv_cl_internal). This is the one place the Tensor -- and
    // therefore the answer -- is still in hand.
    const bool out_svm =
      result.getMemoryData() && result.getMemoryData()->isSVM();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda, out_svm)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda, out_svm);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m
        ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb, out_svm)
        : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb, out_svm);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc,
               out_svm);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void multiplyCl(Tensor &input, float const &value) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData<float>();
    unsigned int len = input.size();

    scal_cl(len, value, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data = input.getData<_FP16>();
    unsigned int len = input.size();
    sscal_cl(data, len, value);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void add_i_cl(Tensor &result, Tensor const &input) {

  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
    << result.getName() << " is not allocated";

  // Broadcasting done for the case where batch size vary for both inputs
  // If batch size vary, batch size of input must be 1
  if ((result.getDim() == input.getDim()) ||
      (result.getDim() != input.getDim() && input.batch() == 1 &&
       result.channel() == input.channel() &&
       result.height() == input.height() && result.width() == input.width())) {

    if (result.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *Y = result.getData();
      const float *X = input.getData();

      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        axpy_cl(input.size(), 1.0f, X, Y);
        Y += input.size();
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

      addition_cl(data_input, data_res, size_input, size_res);

#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }

  else {
    throw std::invalid_argument(
      "Error: Broadcasting not supported for these dimensions!");
  }
}

void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = in.getData();
    float *rdata = result.getData();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void copyCl(const Tensor &input, Tensor &result) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    float *rdata = result.getData();

    unsigned int len = input.size();

    copy_cl(len, data, rdata);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, copyCl not supported for FP16");
#endif
  }
}

float nrm2Cl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = nrm2_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, nrm2Cl not supported for FP16");
#endif
  }

  return result;
}

float asumCl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = asum_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, asumCl not supported for FP16");
#endif
  }

  return result;
}

int amaxCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amax_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

int aminCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amin_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

} // namespace nntrainer
