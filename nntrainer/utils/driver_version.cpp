// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   driver_version.cpp
 * @date   18 September 2026
 * @brief  Accelerator driver / runtime version probe and minimum-version gate.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "driver_version.h"

#include "dynamic_library_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#pragma comment(lib, "version.lib")
#endif

namespace nntrainer {

namespace {

/** Candidate names of the CUDA *driver* library (not the toolkit runtime).
 *  cuDriverGetVersion() lives here and, unlike almost every other entry point,
 *  is documented to work before cuInit(), so probing costs no context. */
const char *const kCudaDriverLibs[] = {
#if defined(_WIN32)
  "nvcuda.dll",
#elif defined(__APPLE__)
  "libcuda.dylib",
#else
  "libcuda.so.1",
  "libcuda.so",
#endif
};

/** Candidate names of the CUDA runtime (cudart). Optional: absent on a machine
 *  that has a driver but no toolkit, which is the normal end-user case. */
const char *const kCudaRuntimeLibs[] = {
#if defined(_WIN32)
  "cudart64_12.dll",
  "cudart64_110.dll",
  "cudart64_101.dll",
#elif defined(__APPLE__)
  "libcudart.dylib",
#else
  "libcudart.so",
  "libcudart.so.12",
  "libcudart.so.11.0",
#endif
};

const char *const kOpenClLibs[] = {
#if defined(_WIN32)
  "OpenCL.dll",
#elif defined(__APPLE__)
  "libOpenCL.dylib",
#else
  "libOpenCL.so.1",
  "libOpenCL.so",
#endif
};

/** OpenCL constants, spelled locally so this TU needs no CL headers (it has to
 *  compile on a build configured without OpenCL, and on MSVC). Values are from
 *  the Khronos cl.h and are ABI-frozen. */
constexpr unsigned int kClPlatformName = 0x0902;
constexpr unsigned int kClPlatformVersion = 0x0901;
constexpr unsigned int kClDeviceName = 0x102B;
constexpr unsigned int kClDriverVersion = 0x102D;
constexpr unsigned long long kClDeviceTypeAll = 0xFFFFFFFF;

void *loadFirst(const char *const *names, size_t n, std::string &tried) {
  for (size_t i = 0; i < n; ++i) {
    void *h = DynamicLibraryLoader::loadLibrary(names[i], RTLD_LAZY);
    if (h != nullptr)
      return h;
    if (!tried.empty())
      tried += ", ";
    tried += names[i];
  }
  return nullptr;
}

/** Deliberately never freed: the ICD / driver may have registered atexit
 *  handlers and other parts of the process may already hold it. A probe must
 *  not be able to unload a library out from under the engine. */
void probeCuda(DriverVersionInfo &out) {
  std::string tried;
  void *lib =
    loadFirst(kCudaDriverLibs,
              sizeof(kCudaDriverLibs) / sizeof(kCudaDriverLibs[0]), tried);
  if (lib == nullptr) {
    out.diagnostics += "cuda: no driver library (" + tried + "); ";
    return;
  }

  using cuDriverGetVersion_t = int (*)(int *);
  using cuInit_t = int (*)(unsigned int);
  using cuDeviceGetCount_t = int (*)(int *);

  auto get_version = reinterpret_cast<cuDriverGetVersion_t>(
    DynamicLibraryLoader::loadSymbol(lib, "cuDriverGetVersion"));
  if (get_version == nullptr) {
    out.diagnostics += "cuda: driver library has no cuDriverGetVersion; ";
    return;
  }

  int version = 0;
  if (get_version(&version) != 0 /* CUDA_SUCCESS */) {
    out.diagnostics += "cuda: cuDriverGetVersion failed; ";
    return;
  }
  out.cuda_present = true;
  out.cuda_driver_version = version;

  /** Device count needs cuInit(0). That is heavier than the version query but
   *  still allocates no device memory and creates no context (a context is
   *  cuCtxCreate); it is what tells apart "driver installed" from "driver
   *  installed and a usable GPU is visible". Failure is not an error: a
   *  headless or CUDA_VISIBLE_DEVICES="" machine legitimately has none. */
  auto cu_init =
    reinterpret_cast<cuInit_t>(DynamicLibraryLoader::loadSymbol(lib, "cuInit"));
  auto device_count = reinterpret_cast<cuDeviceGetCount_t>(
    DynamicLibraryLoader::loadSymbol(lib, "cuDeviceGetCount"));
  if (cu_init != nullptr && device_count != nullptr && cu_init(0) == 0) {
    int count = 0;
    if (device_count(&count) == 0)
      out.cuda_device_count = count;
  }

  std::string rt_tried;
  void *rt =
    loadFirst(kCudaRuntimeLibs,
              sizeof(kCudaRuntimeLibs) / sizeof(kCudaRuntimeLibs[0]), rt_tried);
  if (rt != nullptr) {
    using cudaRuntimeGetVersion_t = int (*)(int *);
    auto rt_version = reinterpret_cast<cudaRuntimeGetVersion_t>(
      DynamicLibraryLoader::loadSymbol(rt, "cudaRuntimeGetVersion"));
    int rtv = 0;
    if (rt_version != nullptr && rt_version(&rtv) == 0)
      out.cuda_runtime_version = rtv;
  }
}

void probeOpenCl(DriverVersionInfo &out) {
  std::string tried;
  void *lib =
    loadFirst(kOpenClLibs, sizeof(kOpenClLibs) / sizeof(kOpenClLibs[0]), tried);
  if (lib == nullptr) {
    out.diagnostics += "opencl: no ICD (" + tried + "); ";
    return;
  }

  using clGetPlatformIDs_t = int (*)(unsigned int, void **, unsigned int *);
  using clGetPlatformInfo_t =
    int (*)(void *, unsigned int, size_t, void *, size_t *);
  using clGetDeviceIDs_t =
    int (*)(void *, unsigned long long, unsigned int, void **, unsigned int *);
  using clGetDeviceInfo_t =
    int (*)(void *, unsigned int, size_t, void *, size_t *);

  auto platform_ids = reinterpret_cast<clGetPlatformIDs_t>(
    DynamicLibraryLoader::loadSymbol(lib, "clGetPlatformIDs"));
  auto platform_info = reinterpret_cast<clGetPlatformInfo_t>(
    DynamicLibraryLoader::loadSymbol(lib, "clGetPlatformInfo"));
  auto device_ids = reinterpret_cast<clGetDeviceIDs_t>(
    DynamicLibraryLoader::loadSymbol(lib, "clGetDeviceIDs"));
  auto device_info = reinterpret_cast<clGetDeviceInfo_t>(
    DynamicLibraryLoader::loadSymbol(lib, "clGetDeviceInfo"));
  if (platform_ids == nullptr || platform_info == nullptr ||
      device_ids == nullptr || device_info == nullptr) {
    out.diagnostics += "opencl: ICD is missing core entry points; ";
    return;
  }

  void *platforms[8] = {nullptr};
  unsigned int num_platforms = 0;
  if (platform_ids(8, platforms, &num_platforms) != 0 || num_platforms == 0) {
    out.diagnostics += "opencl: no platform; ";
    return;
  }

  auto str_info = [](auto fn, void *obj, unsigned int param) -> std::string {
    char buf[512] = {0};
    size_t len = 0;
    if (fn(obj, param, sizeof(buf) - 1, buf, &len) != 0)
      return std::string();
    return std::string(buf);
  };

  /** First platform that actually offers a device wins. A machine can carry a
   *  stub ICD entry (the Android/ARM cross build installs one) that enumerates
   *  as a platform and then has nothing behind it. */
  for (unsigned int i = 0; i < num_platforms; ++i) {
    void *devices[8] = {nullptr};
    unsigned int num_devices = 0;
    if (device_ids(platforms[i], kClDeviceTypeAll, 8, devices, &num_devices) !=
          0 ||
        num_devices == 0)
      continue;

    out.opencl_present = true;
    out.opencl_platform_name =
      str_info(platform_info, platforms[i], kClPlatformName);
    out.opencl_platform_version =
      str_info(platform_info, platforms[i], kClPlatformVersion);
    out.opencl_device_name = str_info(device_info, devices[0], kClDeviceName);
    out.opencl_driver_version =
      str_info(device_info, devices[0], kClDriverVersion);
    return;
  }

  out.diagnostics += "opencl: platform present but no device; ";
}

void probeWddm(DriverVersionInfo &out) {
#if defined(_WIN32)
  /** The WDDM user-mode driver version a CUDA app actually runs against is the
   *  file version of nvcuda.dll in the driver store; the CUDA API only reports
   *  the API level it implements. Windows-only, and non-fatal: an empty string
   *  means "not read", never "too old". */
  char path[MAX_PATH] = {0};
  UINT n = GetSystemDirectoryA(path, MAX_PATH);
  if (n == 0 || n >= MAX_PATH)
    return;
  std::string dll = std::string(path) + "\\nvcuda.dll";

  DWORD dummy = 0;
  DWORD size = GetFileVersionInfoSizeA(dll.c_str(), &dummy);
  if (size == 0)
    return;
  std::vector<unsigned char> block(size);
  if (!GetFileVersionInfoA(dll.c_str(), 0, size, block.data()))
    return;

  VS_FIXEDFILEINFO *ffi = nullptr;
  UINT ffi_len = 0;
  if (!VerQueryValueA(block.data(), "\\", reinterpret_cast<LPVOID *>(&ffi),
                      &ffi_len) ||
      ffi == nullptr)
    return;

  char buf[64];
  std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", HIWORD(ffi->dwFileVersionMS),
                LOWORD(ffi->dwFileVersionMS), HIWORD(ffi->dwFileVersionLS),
                LOWORD(ffi->dwFileVersionLS));
  out.wddm_driver_version = buf;
#else
  (void)out;
#endif
}

std::string upperKey(const std::string &key) {
  std::string s;
  s.reserve(key.size());
  for (char c : key)
    s += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

/** A row's value may be written either dotted ("12.0") or already comparable
 *  ("12000000"); anything with a '.' is dotted, anything above the largest
 *  dotted encoding we would ever emit is comparable, and a bare small integer
 *  is read as a CUDA-encoded version (12040) for the cuda rows and as a major
 *  version otherwise. Ambiguity resolved per-key rather than guessed. */
long long parseRequirementValue(const std::string &key,
                                const std::string &value) {
  if (value.find('.') != std::string::npos)
    return parseDottedVersion(value);

  const long long raw = std::strtoll(value.c_str(), nullptr, 10);
  if (raw <= 0)
    return 0;
  if (key.compare(0, 4, "cuda") == 0 && raw < 1000000)
    return cudaVersionToComparable(static_cast<int>(raw));
  if (raw < 1000)
    return raw * 1000000; /* bare major version */
  return raw;
}

void applyOverrides(std::vector<DriverRequirement> &rows) {
  /** File first, environment second: a pack can ship a requirements file and a
   *  bug-report reproduction can still move one row from the shell. */
  if (const char *file = std::getenv("NNTR_DRIVER_MIN_FILE")) {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
      const size_t hash = line.find('#');
      if (hash != std::string::npos)
        line.erase(hash);
      const size_t eq = line.find('=');
      if (eq == std::string::npos)
        continue;
      std::string key = line.substr(0, eq);
      std::string value = line.substr(eq + 1);
      auto trim = [](std::string &s) {
        while (!s.empty() &&
               std::isspace(static_cast<unsigned char>(s.front())))
          s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
          s.pop_back();
      };
      trim(key);
      trim(value);
      for (auto &row : rows) {
        if (row.key == key) {
          row.min_value = parseRequirementValue(key, value);
          row.human = value;
        }
      }
    }
  }

  for (auto &row : rows) {
    const std::string env_name = "NNTR_DRIVER_MIN_" + upperKey(row.key);
    if (const char *value = std::getenv(env_name.c_str())) {
      row.min_value = parseRequirementValue(row.key, value);
      row.human = value;
    }
  }
}

} // namespace

long long parseDottedVersion(const std::string &text) {
  size_t i = 0;
  while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])))
    ++i;
  if (i >= text.size())
    return 0;

  long long part[3] = {0, 0, 0};
  int found = 0;
  while (i < text.size() && found < 3) {
    long long v = 0;
    bool any = false;
    while (i < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[i]))) {
      v = v * 10 + (text[i] - '0');
      if (v > 999999)
        v = 999999;
      ++i;
      any = true;
    }
    if (!any)
      break;
    part[found++] = v;
    if (i < text.size() && text[i] == '.')
      ++i;
    else
      break;
  }

  const long long major = std::min<long long>(part[0], 999);
  const long long minor = std::min<long long>(part[1], 999);
  const long long patch = std::min<long long>(part[2], 999);
  return major * 1000000 + minor * 1000 + patch;
}

long long cudaVersionToComparable(int cuda_encoded) {
  if (cuda_encoded <= 0)
    return 0;
  const long long major = cuda_encoded / 1000;
  const long long minor = (cuda_encoded % 1000) / 10;
  return major * 1000000 + minor * 1000;
}

const std::vector<DriverRequirement> &driverRequirements() {
  static std::mutex mutex;
  static std::vector<DriverRequirement> rows;
  std::lock_guard<std::mutex> lock(mutex);

  /** Built-in defaults. These are DATA, deliberately: each row says what is
   *  known to break below it, and any row can be replaced from a file or the
   *  environment without a rebuild (see driverRequirements() docs). A row with
   *  min_value 0 is report-only and can never fail a load. */
  rows = {
    {"cuda_driver", cudaVersionToComparable(12000), "12.0",
     "the CUDA kernels are built for the 12.x driver ABI; an 11.x driver "
     "either refuses the module or mis-links the cuBLAS workspace"},
    {"cuda_runtime", 0, "(report only)",
     "informational: the toolkit beside the driver, absent on end-user "
     "machines"},
    {"opencl_platform", 2 * 1000000, "OpenCL 2.0",
     "the weight arena uses coarse-grained SVM and the KV cache uses image "
     "reads, neither of which exists in OpenCL 1.2"},
    {"opencl_driver", 0, "(report only)",
     "vendor build number; recorded so a measurement can be attributed, not "
     "gated"},
    {"wddm_driver", 0, "(report only)",
     "Windows display-driver build behind nvcuda.dll; recorded for the WDDM "
     "residency levers"},
  };

  applyOverrides(rows);
  return rows;
}

const DriverVersionInfo &queryDriverVersions(bool force) {
  static std::mutex mutex;
  static DriverVersionInfo info;
  static bool probed = false;
  std::lock_guard<std::mutex> lock(mutex);

  if (probed && !force)
    return info;

  info = DriverVersionInfo();
  probeCuda(info);
  probeOpenCl(info);
  probeWddm(info);
  probed = true;
  return info;
}

std::string formatDriverVersions(const DriverVersionInfo &info) {
  std::ostringstream os;

  if (info.cuda_present) {
    os << "cuda driver " << (info.cuda_driver_version / 1000) << "."
       << ((info.cuda_driver_version % 1000) / 10);
    if (info.cuda_runtime_version > 0)
      os << ", runtime " << (info.cuda_runtime_version / 1000) << "."
         << ((info.cuda_runtime_version % 1000) / 10);
    os << ", devices " << info.cuda_device_count;
  } else {
    os << "cuda driver absent";
  }

  os << " | ";
  if (info.opencl_present) {
    os << "opencl " << info.opencl_platform_version << " ["
       << info.opencl_platform_name << "] device '" << info.opencl_device_name
       << "' driver " << info.opencl_driver_version;
  } else {
    os << "opencl absent";
  }

  if (!info.wddm_driver_version.empty())
    os << " | wddm " << info.wddm_driver_version;
  if (!info.diagnostics.empty())
    os << " | notes: " << info.diagnostics;

  return os.str();
}

bool checkDriverRequirements(const DriverVersionInfo &info,
                             const std::string &engine, std::string &message) {
  message.clear();
  if (engine == "cpu" || engine.empty())
    return true;

  const auto &rows = driverRequirements();
  auto row_of = [&rows](const char *key) -> const DriverRequirement * {
    for (const auto &r : rows)
      if (r.key == key)
        return &r;
    return nullptr;
  };

  std::ostringstream fail;

  if (engine == "cuda") {
    if (!info.cuda_present) {
      message =
        "the 'cuda' backend was requested but no CUDA driver library could be "
        "loaded (" +
        info.diagnostics +
        "). Install the NVIDIA driver, or select the 'gpu' (OpenCL) or 'cpu' "
        "backend.";
      return false;
    }
    if (info.cuda_device_count == 0) {
      message = "the 'cuda' backend was requested and the driver loaded (" +
                formatDriverVersions(info) +
                "), but it reports 0 visible devices. Check that a GPU is "
                "present and that CUDA_VISIBLE_DEVICES is not empty.";
      return false;
    }
    if (const auto *row = row_of("cuda_driver")) {
      const long long found = cudaVersionToComparable(info.cuda_driver_version);
      if (row->min_value > 0 && found < row->min_value)
        fail << "CUDA driver " << (info.cuda_driver_version / 1000) << "."
             << ((info.cuda_driver_version % 1000) / 10) << " is below the "
             << "required " << row->human << " (" << row->reason << "). ";
    }
    if (const auto *row = row_of("cuda_runtime")) {
      const long long found =
        cudaVersionToComparable(info.cuda_runtime_version);
      if (row->min_value > 0 && info.cuda_runtime_version > 0 &&
          found < row->min_value)
        fail << "CUDA runtime " << (info.cuda_runtime_version / 1000) << "."
             << ((info.cuda_runtime_version % 1000) / 10) << " is below the "
             << "required " << row->human << " (" << row->reason << "). ";
    }
    if (const auto *row = row_of("wddm_driver")) {
      if (row->min_value > 0 && !info.wddm_driver_version.empty() &&
          parseDottedVersion(info.wddm_driver_version) < row->min_value)
        fail << "the Windows display driver " << info.wddm_driver_version
             << " is below the required " << row->human << " (" << row->reason
             << "). ";
    }
  } else if (engine == "gpu") {
    if (!info.opencl_present) {
      message =
        "the 'gpu' (OpenCL) backend was requested but no OpenCL device could "
        "be enumerated (" +
        info.diagnostics +
        "). Install the vendor OpenCL runtime, or select the 'cpu' backend.";
      return false;
    }
    if (const auto *row = row_of("opencl_platform")) {
      const long long found = parseDottedVersion(info.opencl_platform_version);
      if (row->min_value > 0 && found < row->min_value)
        fail << "the OpenCL platform reports '" << info.opencl_platform_version
             << "', below the required " << row->human << " (" << row->reason
             << "). ";
    }
    if (const auto *row = row_of("opencl_driver")) {
      const long long found = parseDottedVersion(info.opencl_driver_version);
      if (row->min_value > 0 && found < row->min_value)
        fail << "the OpenCL driver reports '" << info.opencl_driver_version
             << "', below the required " << row->human << " (" << row->reason
             << "). ";
    }
  }

  const std::string text = fail.str();
  if (text.empty())
    return true;

  message = text + "Found: " + formatDriverVersions(info) +
            ". Override a row with NNTR_DRIVER_MIN_<ROW> (e.g. "
            "NNTR_DRIVER_MIN_CUDA_DRIVER=0 to skip this check) once you have "
            "confirmed the configuration yourself.";
  return false;
}

} // namespace nntrainer
