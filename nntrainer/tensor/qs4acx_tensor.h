// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4acx_tensor.h
 * @date	02 October 2026
 * @brief	QS4ACX_Tensor: per-output-channel asymmetric 4-bit weight, stored
 *          as the QS4CX byte stream followed by one fp32 zero point per
 *          output channel.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * Layout (height() = K = input, width() = N = output, like QS4CX):
 *
 *   [ N rows x ceil(K/2) bytes of nibbles ][ fp32 scale[N] ][ fp32 zp[N] ]
 *
 * The first N * ceil(K/2) + 4N bytes are exactly a QS4CX record: nibble of
 * row n, input k is byte n * ceil(K/2) + k/2 (even k low nibble, odd k high),
 * u = c + 8 with the signed code c in [-8, 7], and scale[n] is the step.
 * The zero point is an integer z in [0, 15] stored as fp32, and
 *
 *   w[k][n] = scale[n] * (c + 8 - zp[n])
 *   y[m][n] = scale[n] * sum_k c[n][k] x[m][k] + corr[n] * sum_k x[m][k]
 *   corr[n] = scale[n] * (8 - zp[n])
 *
 * so a QS4CX kernel plus the rank-1 correction corr x rowsum(x) computes the
 * asymmetric product; with zp == 8 everywhere the tensor is a QS4CX tensor.
 */

#ifndef __QS4ACX_TENSOR_H__
#define __QS4ACX_TENSOR_H__
#ifdef __cplusplus

#include <mutex>
#include <vector>

#include <qs4cx_tensor.h>

namespace nntrainer {

/**
 * @class QS4ACX_Tensor
 * @brief per-output-channel asymmetric 4-bit weight (QS4CX record + zp[N])
 */
class QS4ACX_Tensor : public QS4CX_Tensor {
public:
  /**
   * @brief Basic constructor
   */
  QS4ACX_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new QS4ACX_Tensor
   * @param d dimension (batch and channel must be 1)
   * @param alloc_now allocate now
   * @param init initializer
   * @param name name
   */
  QS4ACX_Tensor(const TensorDim &d, bool alloc_now,
                Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new QS4ACX_Tensor from a buffer of size() bytes
   */
  QS4ACX_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct from a TensorBase
   */
  QS4ACX_Tensor(TensorBase &rhs) : QS4CX_Tensor(rhs) {
    setQs4cxRecordPadded(false);
  }

  /**
   * @brief bytes of a QS4ACX record: QS4CX record + 4 N
   */
  static size_t recordBytes(size_t K, size_t N) {
    return N * ((K + 1) / 2) + 2 * N * sizeof(float);
  }

  /**
   * @copydoc TensorBase::size()
   * @note BYTES: QS4CX record plus N fp32 zero points
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override { return size(); }

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /**
   * @brief fp32 zero points, width() of them
   */
  const float *getZeroPointFp32() const;

  /**
   * @brief corr[n] = scale[n] * (8 - zp[n]), computed once from the loaded
   *        record (the record must not change afterwards)
   */
  const float *getCorrection() const;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override;

private:
  std::string getStringDataType() const override { return "QS4ACX"; }

  mutable std::once_flag corr_once;
  mutable std::vector<float> corr;
};

/**
 * @brief REFERENCE QS4ACX GEMM: C[M, N] = A[M, K] * W[K, N] (fp32, no
 *        activation quantization), threaded over output channels.
 *
 * Per output: acc = fma(c[n][k], x[m][k], acc) in ascending k (one fp32
 * accumulator from +0), rowsum(x[m]) = plain fp32 sum in ascending k, and
 * C = fma(corr[n], rowsum, scale[n] * acc). With corr == 0 (zp == 8) this is
 * exactly scale[n] * acc, the symmetric QS4CX product. Deterministic: every
 * output is computed by one task in a fixed order.
 *
 * @param M activation rows
 * @param N output channels
 * @param K input size
 * @param A activation, row stride lda
 * @param lda activation row stride
 * @param W QS4ACX record
 * @param corr correction per output channel (QS4ACX_Tensor::getCorrection),
 *        or nullptr to derive it from the record's scale and zp
 * @param C output, row stride ldc
 * @param ldc output row stride
 */
void qs4acx_gemm_ref_fp32(size_t M, size_t N, size_t K, const float *A,
                          size_t lda, const void *W, const float *corr,
                          float *C, size_t ldc);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QS4ACX_TENSOR_H__ */
