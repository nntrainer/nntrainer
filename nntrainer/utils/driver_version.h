// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   driver_version.h
 * @date   18 September 2026
 * @brief  Accelerator driver / runtime version probe and minimum-version gate.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A GPU backend that loads and then produces wrong numbers, or that
 *          silently serializes, is almost always a driver too old for the
 *          kernels the engine compiles. This probe answers "which driver am I
 *          actually on" before the first kernel is built, and compares it with
 *          a *data* table of minimums (overridable from the environment or a
 *          file) rather than a policy baked into the code.
 *
 *          Every probe here is side-effect free and SDK-free: the CUDA driver
 *          and the OpenCL ICD are reached through DynamicLibraryLoader, so this
 *          TU compiles and runs identically on a build without CUDA headers,
 *          without OpenCL headers, and on MSVC. A machine with no driver at all
 *          reports "absent", never a crash.
 */

#ifndef __NNTRAINER_DRIVER_VERSION_H__
#define __NNTRAINER_DRIVER_VERSION_H__

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @brief What the probe found on this machine. Numeric CUDA versions use the
 *        CUDA encoding (1000 * major + 10 * minor), e.g. 12040 == 12.4.
 *        String versions are whatever the vendor reported, verbatim.
 */
struct DriverVersionInfo {
  bool cuda_present = false;    /**< CUDA driver library loadable */
  int cuda_driver_version = 0;  /**< cuDriverGetVersion() */
  int cuda_runtime_version = 0; /**< cudaRuntimeGetVersion(), 0 if no cudart */
  int cuda_device_count = 0;    /**< visible CUDA devices (0 = none) */

  bool opencl_present = false; /**< OpenCL ICD loadable with >=1 device */
  std::string opencl_platform_version; /**< CL_PLATFORM_VERSION */
  std::string opencl_driver_version;   /**< CL_DRIVER_VERSION */
  std::string opencl_device_name;      /**< CL_DEVICE_NAME */
  std::string opencl_platform_name;    /**< CL_PLATFORM_NAME */

  /** Windows display-driver (WDDM) version of the NVIDIA user-mode driver,
   *  read from nvcuda.dll's file version. Empty everywhere else. */
  std::string wddm_driver_version;

  std::string diagnostics; /**< why a probe came back empty, if it did */
};

/**
 * @brief One row of the minimum-version table. This is data: the built-in rows
 *        are defaults, and both a file and the environment can replace any row
 *        without recompiling.
 */
struct DriverRequirement {
  std::string key;     /**< "cuda_driver" | "cuda_runtime" | "opencl_platform" |
                            "opencl_driver" | "wddm_driver" */
  long long min_value; /**< comparable form; 0 = report only, never fail */
  std::string human;   /**< how to print the minimum to a user */
  std::string reason;  /**< what breaks below it */
};

/**
 * @brief Parse a leading dotted version ("24.35.30872", "OpenCL 3.0 NEO",
 *        "555.42.02") into a comparable integer
 *        major * 1000000 + minor * 1000 + patch. Returns 0 when no number is
 *        found. Components are clamped to 999 so a huge build number cannot
 *        carry into the minor field.
 */
long long parseDottedVersion(const std::string &text);

/**
 * @brief Encode a CUDA-style version (12040) in the same comparable form as
 *        parseDottedVersion, so one table can hold both kinds of row.
 */
long long cudaVersionToComparable(int cuda_encoded);

/**
 * @brief The built-in minimum table, already merged with the overrides.
 * @details Override precedence, lowest first:
 *          1. the built-in rows below,
 *          2. a file named by NNTR_DRIVER_MIN_FILE — one `key=version` per
 *             line, `#` comments allowed, version either dotted ("12.0") or
 *             already comparable,
 *          3. NNTR_DRIVER_MIN_<KEY> in the environment (KEY upper-cased, e.g.
 *             NNTR_DRIVER_MIN_CUDA_DRIVER=12.2). Setting a row to 0 disables
 *             that check.
 *          Re-read on every call, so a harness can change a row between runs.
 */
const std::vector<DriverRequirement> &driverRequirements();

/**
 * @brief Probe the machine. Cached after the first call (drivers do not change
 *        inside a process); pass force=true to re-probe.
 */
const DriverVersionInfo &queryDriverVersions(bool force = false);

/**
 * @brief One-line-per-item human report of @p info, for logs and for
 *        get_last_error style surfaces.
 */
std::string formatDriverVersions(const DriverVersionInfo &info);

/**
 * @brief Check @p info against driverRequirements() for one engine.
 * @param engine nntrainer engine string: "cuda", "gpu" (OpenCL) or "cpu".
 *               "cpu" always passes.
 * @param[out] message empty on success; otherwise a complete, actionable
 *               sentence naming the found version, the required version and
 *               what fails below it.
 * @return true when every applicable row is satisfied (or is report-only).
 */
bool checkDriverRequirements(const DriverVersionInfo &info,
                             const std::string &engine, std::string &message);

} // namespace nntrainer

#endif /* __NNTRAINER_DRIVER_VERSION_H__ */
