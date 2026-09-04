// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer_manager.h
 * @date    01 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#ifndef __OPENCL_BUFFER_MANAGER_H__
#define __OPENCL_BUFFER_MANAGER_H__

#include <string>

#include <opencl_buffer.h>
#include <opencl_context_manager.h>

#include <nntrainer_log.h>

#include "singleton.h"

namespace nntrainer {

/**
 * @class ClBufferManager contains Buffer object management
 * @brief Support for Buffer management
 */

class ClBufferManager : public Singleton<ClBufferManager> {

private:
  /**
   * @brief OpenCl context global instance
   */
  opencl::ContextManager &context_inst_ = opencl::ContextManager::Global();

  /**
   * @brief Buffer size in bytes preset (32 mebibytes)
   */
  const size_t buffer_size_bytes = 1024 * 8192 * sizeof(float);
  const size_t unused_buffer_bytes = sizeof(float);
  const unsigned int max_qs = 3;

  /// @note this size might be changed
  const size_t scale_q4_0_size =
    3072 * (8192 / 32) * 2;                   /** buffer size of quants */
  const size_t quant_q4_0_size = 3072 * 8192; /** buffer size of scales */

  opencl::Buffer *inBufferA = nullptr;
  opencl::Buffer *inBufferB = nullptr;
  opencl::Buffer *inBufferC = nullptr;
  opencl::Buffer *outBufferA = nullptr;
  opencl::Buffer *outBufferB = nullptr;

  void *data_input = nullptr;
  std::vector<void *> scale_vec;
  std::vector<void *> quant_vec;
  std::vector<void *> output_vec;

public:
  /**
   * @brief Does nothing, and no longer needs calling.
   *
   * Every region is allocated on its first use now, in the getters below, so
   * a process that never runs a block-quantized GEMM pays nothing for them.
   * The name outlived the behaviour: nothing here initialises a buffer any
   * more, and the OpenCL context bring-up no longer calls it.
   *
   * Kept only because this header is installed, so dropping the symbol would
   * break anything compiled against the previous package. New code should not
   * call it.
   */
  void initBuffers();

  /**
   * @brief Get read only inBufferA.
   * @return opencl::Buffer*
   */
  opencl::Buffer *getInBufferA();

  /**
   * @brief Get read only inBufferB.
   * @return opencl::Buffer*
   */
  opencl::Buffer *getInBufferB();

  /**
   * @brief Get read only inBufferC.
   * @return opencl::Buffer*
   */
  opencl::Buffer *getInBufferC();

  /**
   * @brief Get read-write outBufferA.
   * @return opencl::Buffer*
   */
  opencl::Buffer *getOutBufferA();

  /**
   * @brief Get read-write outBufferB.
   * @return opencl::Buffer*
   */
  opencl::Buffer *getOutBufferB();

  /**
   * @brief Get the SVM pointer to data_input
   */
  void *getSVMInput();

  /**
   * @brief Get the SVM pointer to data_input
   *
   * @note remove this when fp16 is enabled on Windows
   */
  void *getSVMOutput(unsigned int idx = 0);

  /**
   * @brief Get the SVM pointer to data_input
   */
  void *getSVMScale(unsigned int idx = 0);

  /**
   * @brief Get the SVM pointer to data_input
   */
  void *getSVMQuant(unsigned int idx = 0);

  /**
   * @brief Destructor. Deliberately does not release the OpenCL regions --
   *        see the note on the definition.
   */
  ~ClBufferManager();
};
} // namespace nntrainer

#endif /* __OPENCL_BUFFER_MANAGER_H__ */
