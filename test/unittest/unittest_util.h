// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_util.h
 * @brief Shared utility functions for unit tests.
 */

#ifndef NNTRAINER_UNITTEST_UTIL_H
#define NNTRAINER_UNITTEST_UTIL_H

#include <vector>
#include <cstddef>
#include <random>

namespace nntrainer {

// Generate a random vector of given size and range.
// The template type T is expected to be convertible from float.
// This function is used across many unit tests.

template <typename T, bool random_init = false>
std::vector<T> generate_random_vector(size_t size, float min_val = -1.F,
                                      float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

// Allocate SVM memory using the OpenCL context.
void *allocateSVM(size_t size_bytes);

// Release SVM memory.
void freeSVM(void *ptr);

} // namespace nntrainer

#endif // NNTRAINER_UNITTEST_UTIL_H
