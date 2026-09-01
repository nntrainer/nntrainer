// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_module.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for runtime kernel compilation (NVRTC) + module load.
 *          Peer of nntrainer::opencl::Program: compiles a .cu source STRING via
 *          NVRTC to PTX, caches the PTX on disk keyed by device signature + a
 *          hash of (source, options), then loads it via the Driver API.
 */

#ifndef __CUDA_MODULE_H__
#define __CUDA_MODULE_H__

#include <cstddef>
#include <string>

#include <cuda.h>

namespace nntrainer::cuda {

/**
 * @class Module
 * @brief Owns a CUmodule compiled from a .cu source string via NVRTC.
 */
class Module {
public:
  /**
   * @brief Compile @p source via NVRTC (with on-disk PTX cache) and load it.
   *
   * @param source       full .cu source string
   * @param name_for_log short tag used in NVRTC file name / logs
   * @param options      extra space-separated NVRTC options (arch is added
   *                     automatically from the active device)
   * @return true on success
   */
  bool CreateModuleFromSource(const std::string &source,
                              const std::string &name_for_log,
                              const std::string &options = "");

  /**
   * @brief Get the loaded module handle
   */
  CUmodule GetModule() const { return module_; }

  /**
   * @brief true if a module is loaded
   */
  bool valid() const { return module_ != nullptr; }

  /**
   * @brief Hash of (code, options) used as the PTX cache key.
   */
  static std::size_t GetKernelHash(const std::string &code,
                                   const std::string &options);

  /**
   * @brief Unload the module.
   */
  ~Module();

private:
  bool compileWithNVRTC(const std::string &source, const std::string &options,
                        std::string &ptx_out, const std::string &log_tag);

  CUmodule module_{nullptr};
};

} // namespace nntrainer::cuda

#endif // __CUDA_MODULE_H__
