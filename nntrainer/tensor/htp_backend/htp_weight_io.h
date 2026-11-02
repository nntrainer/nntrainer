// SPDX-License-Identifier: Apache-2.0
/** @file htp_weight_io.h
 * @brief Shared checked HTP weight reads.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_WEIGHT_IO_H
#define NNTR_HTP_WEIGHT_IO_H
#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <weight_file_reader.h>
namespace nntrainer {
inline int preadAll(int fd, void *dst, size_t len, uint64_t off,
                    void *tail = nullptr, size_t tail_bytes = 0) {
  try {
    return weightPreadSplit(fd, dst, len, tail, tail_bytes, off);
  } catch (const std::bad_alloc &) {
    return ENOMEM;
  } catch (const std::invalid_argument &) {
    return EINVAL;
  } catch (...) {
    return EIO;
  }
}
inline void throwPread(int rc, const char *what) {
  if (rc == 0)
    return;
  if (rc < 0) {
    throw std::runtime_error(std::string("pread(") + what +
                             ") hit end of file: the model file is "
                             "shorter than its weights");
  }
  throw std::runtime_error(std::string("pread(") + what +
                           ") failed: " + std::strerror(rc));
}
} // namespace nntrainer
#endif
