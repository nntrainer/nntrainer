// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4acx_tensor.cpp
 * @date	02 October 2026
 * @brief	QS4ACX_Tensor: per-output-channel asymmetric 4-bit weight.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>

#include <nntrainer_error.h>
#include <qs4acx_tensor.h>
#include <thread_manager.h>

namespace nntrainer {

QS4ACX_Tensor::QS4ACX_Tensor(std::string name_, Tformat fm) :
  QS4CX_Tensor(name_, fm) {
  // the QS4ACX record is laid out at the trimmed QS4CX stride, always
  setQs4cxRecordPadded(false);
}

QS4ACX_Tensor::QS4ACX_Tensor(const TensorDim &d, bool alloc_now,
                             Initializer init, std::string name) :
  QS4CX_Tensor(d, false, init, name) {
  setQs4cxRecordPadded(false);
  // allocate here: inside the base constructor size() is still the QS4CX one
  if (alloc_now)
    allocate();
}

QS4ACX_Tensor::QS4ACX_Tensor(const TensorDim &d, const void *buf) :
  QS4ACX_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0 && buf != nullptr)
    std::memcpy(getData(), buf, size());
}

size_t QS4ACX_Tensor::size() const { return recordBytes(height(), width()); }

QScheme QS4ACX_Tensor::q_scheme() const { return QScheme::QS4ACX; }

const float *QS4ACX_Tensor::getZeroPointFp32() const {
  if (getData() == nullptr)
    return nullptr;
  return reinterpret_cast<const float *>(
    static_cast<const uint8_t *>(getData()) + width() * ((height() + 1) / 2) +
    width() * sizeof(float));
}

const float *QS4ACX_Tensor::getCorrection() const {
  std::call_once(corr_once, [this]() {
    const size_t N = width();
    const float *scale = reinterpret_cast<const float *>(
      static_cast<const uint8_t *>(getData()) + N * ((height() + 1) / 2));
    const float *zp = getZeroPointFp32();
    corr.resize(N);
    for (size_t n = 0; n < N; ++n)
      corr[n] = scale[n] * (8.0f - zp[n]);
  });
  return corr.data();
}

void QS4ACX_Tensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[QS4ACX data print skipped]" << std::endl;
}

void qs4acx_gemm_ref_fp32(size_t M, size_t N, size_t K, const float *A,
                          size_t lda, const void *W, const float *corr,
                          float *C, size_t ldc) {
  const size_t stride = (K + 1) / 2;
  const uint8_t *nib = static_cast<const uint8_t *>(W);
  const float *scale = reinterpret_cast<const float *>(nib + N * stride);
  std::vector<float> own_corr;
  if (corr == nullptr) {
    const float *zp = scale + N;
    own_corr.resize(N);
    for (size_t n = 0; n < N; ++n)
      own_corr[n] = scale[n] * (8.0f - zp[n]);
    corr = own_corr.data();
  }

  std::vector<float> rowsum(M);
  for (size_t m = 0; m < M; ++m) {
    const float *x = A + m * lda;
    float s = 0.0f;
    for (size_t k = 0; k < K; ++k)
      s += x[k];
    rowsum[m] = s;
  }

  auto run_channels = [&](size_t n0, size_t n1) {
    std::vector<float> w(K);
    for (size_t n = n0; n < n1; ++n) {
      const uint8_t *row = nib + n * stride;
      for (size_t k = 0; k < K; ++k) {
        const uint8_t b = row[k / 2];
        const int u = (k % 2 == 0) ? (b & 0x0F) : (b >> 4);
        w[k] = static_cast<float>(u - 8);
      }
      for (size_t m = 0; m < M; ++m) {
        const float *x = A + m * lda;
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k)
          acc = std::fma(w[k], x[k], acc);
        C[m * ldc + n] = std::fma(corr[n], rowsum[m], scale[n] * acc);
      }
    }
  };

  constexpr size_t chunk = 64;
  const size_t tasks = (N + chunk - 1) / chunk;
  ThreadManager::Global().parallel_for(0, tasks, [&](size_t t) {
    run_channels(t * chunk, std::min(N, (t + 1) * chunk));
  });
}

} // namespace nntrainer
