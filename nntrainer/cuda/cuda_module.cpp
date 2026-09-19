// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_module.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVRTC compile + PTX disk-cache + Driver-API module load.
 */

#include "cuda_module.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"

#include <cstdlib>
#include <filesystem>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <vector>

namespace nntrainer::cuda {

std::size_t Module::GetKernelHash(const std::string &code,
                                  const std::string &options) {
  return std::hash<std::string>{}(code + std::string("\x01") + options);
}

/**
 * @brief Directory for the compiled-kernel disk cache.
 *
 * The cached image is handed straight to cuModuleLoadData, so whoever can
 * write a file here can get arbitrary machine code executed on the GPU inside
 * this process. It therefore has to land in a directory only this user can
 * write. HOME is unset on Windows and in plenty of Linux service and container
 * environments, and the old fallback of "$HOME or /tmp" put the cache in a
 * world-writable place exactly there. Resolve a per-user location per platform
 * instead, and return "" when there is none -- the caller then compiles every
 * time rather than trusting a directory it cannot vouch for.
 */
static std::string cacheDir() {
  if (const char *e = getenv("NNTR_CUDA_CACHE"))
    return e;
#ifdef _WIN32
  for (const char *v : {"LOCALAPPDATA", "USERPROFILE"}) {
    if (const char *base = getenv(v))
      return std::string(base) + "\\nntrainer\\cuda_kernel_cache";
  }
  return std::string();
#else
  if (const char *xdg = getenv("XDG_CACHE_HOME"))
    return std::string(xdg) + "/nntrainer_cuda";
  if (const char *home = getenv("HOME"))
    return std::string(home) + "/.cache/nntrainer_cuda";
  return std::string();
#endif
}

/**
 * @brief True if the directory exists and is writable by anyone but its owner.
 *
 * A cache directory another user can write into is a code-execution channel,
 * not merely a stale-data risk, so a directory that fails this test is
 * refused rather than repaired: repairing it would race whoever created it.
 */
static bool dirIsUnsafe(const std::string &path) {
#ifdef _WIN32
  (void)path;
  return false; // the per-user roots above are already ACL'd to this user
#else
  struct stat st;
  if (::stat(path.c_str(), &st) != 0)
    return false; // does not exist yet; makeDirs creates it 0700 below
  if (!S_ISDIR(st.st_mode))
    return true;
  if (st.st_uid != ::geteuid())
    return true;
  return (st.st_mode & (S_IWGRP | S_IWOTH)) != 0;
#endif
}

/// best-effort recursive mkdir (mkdir -p semantics)
static void makeDirs(const std::string &path) {
  // std::filesystem::create_directories creates every missing parent (the
  // recursive case the hand-rolled loop handled) and is portable. Best-effort:
  // swallow errors via the error_code overload (dir may already exist or be
  // uncreatable) — matches the old ignore-return behaviour.
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
#ifndef _WIN32
  // Owner-only: see cacheDir(). create_directories applies the process umask,
  // which cannot be relied on to clear the group and other bits.
  std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace, ec);
#endif
}

static bool readFile(const std::string &p, std::string &out) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

static void writeFile(const std::string &p, const std::string &data) {
  std::ofstream f(p, std::ios::binary);
  if (f)
    f.write(data.data(), (std::streamsize)data.size());
}

// Default to CUBIN (native SASS for the device's real arch) instead of PTX so
// cuModuleLoadData loads machine code directly and skips the PTX->SASS JIT.
// That JIT rejects PTX whose ISA version exceeds what the driver knows -- e.g.
// a CUDA 13.3 NVRTC feeding a driver that only advertises CUDA 13.1
// ("cuModuleLoadData: the provided PTX was compiled with an unsupported
// toolchain"), which blocked every kernel on a Windows box with a
// slightly-older driver. A cubin for a driver-supported arch (Blackwell sm_120
// here) has no ISA-version gate and also loads faster (no JIT).
// GetComputeArch() returns the DEVICE's real cc, so the SASS always matches the
// current GPU -- no portability loss vs PTX (we compile per-device at runtime
// anyway). NNTR_CUDA_PTX forces the legacy PTX path.
static bool useCubin() {
  static const bool v = std::getenv("NNTR_CUDA_PTX") == nullptr;
  return v;
}

bool Module::compileWithNVRTC(const std::string &source,
                              const std::string &options, std::string &ptx_out,
                              const std::string &log_tag) {
  nvrtcProgram prog = nullptr;
  if (!nvrtcCheck(nvrtcCreateProgram(&prog, source.c_str(),
                                     (log_tag + ".cu").c_str(), 0, nullptr,
                                     nullptr),
                  "nvrtcCreateProgram"))
    return false;

  // compute_XY -> sm_XY for the cubin path (real SASS target); PTX keeps
  // virtual.
  std::string archname = ContextManager::Global().GetComputeArch();
  if (useCubin()) {
    const std::string cprefix = "compute_";
    if (archname.compare(0, cprefix.size(), cprefix) == 0)
      archname.replace(0, cprefix.size(), "sm_");
  }
  std::string arch = "--gpu-architecture=" + archname;
  std::vector<std::string> extra;
  {
    std::stringstream ss(options);
    std::string tok;
    while (ss >> tok)
      extra.push_back(tok);
  }
  std::vector<const char *> opts;
  opts.push_back(arch.c_str());
  for (const auto &s : extra)
    opts.push_back(s.c_str());

  nvrtcResult r = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  size_t logsz = 0;
  nvrtcGetProgramLogSize(prog, &logsz);
  if (logsz > 1) {
    std::string log(logsz, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    ml_logd("[NVRTC] %s log:\n%s", log_tag.c_str(), log.c_str());
  }
  if (r != NVRTC_SUCCESS) {
    ml_loge("[NVRTC] compile failed for %s", log_tag.c_str());
    nvrtcDestroyProgram(&prog);
    return false;
  }

  // ptx_out carries the loadable image (cubin bytes or PTX text) either way.
  if (useCubin()) {
    size_t cubinsz = 0;
    if (!nvrtcCheck(nvrtcGetCUBINSize(prog, &cubinsz), "nvrtcGetCUBINSize")) {
      nvrtcDestroyProgram(&prog);
      return false;
    }
    ptx_out.resize(cubinsz);
    nvrtcGetCUBIN(prog, &ptx_out[0]);
  } else {
    size_t ptxsz = 0;
    if (!nvrtcCheck(nvrtcGetPTXSize(prog, &ptxsz), "nvrtcGetPTXSize")) {
      nvrtcDestroyProgram(&prog);
      return false;
    }
    ptx_out.resize(ptxsz);
    nvrtcGetPTX(prog, &ptx_out[0]);
  }
  nvrtcDestroyProgram(&prog);
  return true;
}

bool Module::CreateModuleFromSource(const std::string &source,
                                    const std::string &name_for_log,
                                    const std::string &options) {
  ContextManager::Global().EnsureCurrent();

  std::string dir = cacheDir();
  if (!dir.empty() && dirIsUnsafe(dir)) {
    ml_logw("[CUDA] kernel cache directory %s is not owner-only; compiling "
            "without a cache",
            dir.c_str());
    dir.clear();
  }
  std::string sig = ContextManager::Global().GetDeviceSignature();
  for (auto &c : sig)
    if (c == '|' || c == ' ' || c == '/')
      c = '_';
  const std::string key = dir.empty()
                            ? std::string()
                            : dir + "/" + sig + "_" +
                                std::to_string(GetKernelHash(source, options)) +
                                (useCubin() ? ".cubin" : ".ptx");

  std::string ptx;
  bool have_cache = !key.empty() && readFile(key, ptx) && !ptx.empty();
  if (!have_cache) {
    if (!compileWithNVRTC(source, options, ptx, name_for_log))
      return false;
    if (!key.empty()) {
      makeDirs(dir);
      writeFile(key, ptx);
    }
  }

  CUresult lr = cuModuleLoadData(&module_, ptx.c_str());
  if (lr != CUDA_SUCCESS && have_cache) {
    // cached PTX may be stale (driver update); recompile once and retry.
    ml_logw("[CUDA] cached PTX load failed; recompiling %s",
            name_for_log.c_str());
    if (!compileWithNVRTC(source, options, ptx, name_for_log))
      return false;
    writeFile(key, ptx);
    lr = cuModuleLoadData(&module_, ptx.c_str());
  }
  return cuCheck(lr, "cuModuleLoadData");
}

Module::~Module() {
  if (module_) {
    cuModuleUnload(module_);
    module_ = nullptr;
  }
}

} // namespace nntrainer::cuda
