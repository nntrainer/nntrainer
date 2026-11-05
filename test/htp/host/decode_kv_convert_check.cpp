// SPDX-License-Identifier: Apache-2.0
/** @file decode_kv_convert_check.cpp
 * @brief Independent row-major oracle for the DSP checkpoint layout bridge.
 */
#include <algorithm>
#include <htp_decode_kv_convert.h>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
  for (unsigned rows : {1u, 63u, 64u, 65u, 129u})
    for (unsigned dim : {1u, 7u, 8u, 17u, 256u, 512u}) {
      constexpr unsigned heads = 2;
      const size_t width = heads * dim;
      const size_t padded = (rows + 63u) / 64u * 64u;
      std::vector<int8_t> k(padded * width, 127), v(padded * width, -128);
      std::vector<float> sk(width), sv(width);
      std::vector<uint16_t> expected_k(rows * width), expected_v(rows * width);
      for (unsigned h = 0; h < heads; ++h)
        for (unsigned d = 0; d < dim; ++d) {
          const size_t col = size_t(h) * dim + d;
          sk[col] = (col + 1u) * 0.0000131f;
          sv[col] = (col + 3u) * 0.00117f;
          for (unsigned r = 0; r < rows; ++r) {
            const int8_t x = int((r * 19u + col * 7u) % 256u) - 128;
            const int8_t y = int((r * 13u + col * 29u) % 256u) - 128;
            k[((size_t(h) * (padded / 64u) + r / 64u) * dim + d) * 64u +
              r % 64u] = x;
            v[(size_t(h) * padded + r) * dim + d] = y;
            expected_k[size_t(r) * width + col] =
              nntrainer::compute_fp32_to_fp16(float(x) * sk[col]);
            expected_v[size_t(r) * width + col] =
              nntrainer::compute_fp32_to_fp16(float(y) * sv[col]);
          }
        }
      for (unsigned first : {0u, rows / 2u, rows - 1u}) {
        std::vector<uint16_t> actual_k((rows - first) * width),
          actual_v((rows - first) * width);
        nntrainer::decodeKVToPrefill(k.data(), v.data(), sk.data(), sv.data(),
                                     rows, heads, dim, actual_k.data(),
                                     actual_v.data(), first);
        if (!std::equal(actual_k.begin(), actual_k.end(),
                        expected_k.begin() + first * width) ||
            !std::equal(actual_v.begin(), actual_v.end(),
                        expected_v.begin() + first * width))
          throw std::runtime_error(
            "Decode INT8 to prefill FP16 layout mismatch");
      }
    }
  std::cout << "Decode KV to prefill: tile/head/row tails and FP16 bits PASS\n";
}
