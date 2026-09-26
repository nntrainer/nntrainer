// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    footprint_sampler.h
 * @date    07 Sep 2026
 * @brief   Honest peak footprint of THIS run, sampled while it runs
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * @details `getPeakMemoryKb()` (performance_metrics.h) reports
 * getrusage(RUSAGE_SELF).ru_maxrss. Two things are wrong with that as a
 * number to show a user:
 *
 *  1. It does not include the accelerator. On Adreno every clSVMAlloc /
 *     clCreateBuffer is a kgsl allocation that never appears in the process
 *     RSS; on the NPU the weights live in dma-buf. Both are the bulk of the
 *     footprint, and ru_maxrss is blind to them.
 *  2. It is a lifetime high-water the kernel never lowers, so from the second
 *     request onwards it is not "this run's peak" but "the largest this worker
 *     has ever been", model load included.
 *
 * This samples the quantity the measurement harness calls the honest
 * footprint, from the inside and for one run at a time:
 *
 *     peak = max over t of [ RssAnon(self) + accelerator bytes(self) ]
 *
 * RssAnon rather than VmRSS because the accelerator term already counts the
 * mappings the driver hands back (clSVMAlloc shows up in RssFile as well as
 * in kgsl); adding both would count the weights twice. The accelerator term
 * is the larger of the GPU ledger and the dma-buf total for the same reason
 * -- one run uses one of them, and adding them would double count a mirror.
 *
 * The GPU term comes from nntrainer's own allocation ledger and not from
 * /sys/class/kgsl/kgsl/proc/<pid>/gpumem, because that node is Permission
 * denied to the shell uid and to the application uid alike on the Android
 * devices this was measured on. The global page_alloc counter is readable but
 * is not this process's alone. Counting from the inside is the only
 * per-process option, and it closes to 98.7 % of the kgsl delta.
 */

#ifndef __CAUSAL_LM_FOOTPRINT_SAMPLER_H__
#define __CAUSAL_LM_FOOTPRINT_SAMPLER_H__

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <dirent.h>
#endif

#include <performance_metrics.h>

#if defined(ENABLE_OPENCL)
namespace nntrainer::opencl {
/// Declared rather than included, the same way causal_lm.cpp declares the
/// ledger's dump: this header sits in the application and must not pull the
/// OpenCL backend's private headers in behind it.
size_t clMemAcctLiveBytes();
} // namespace nntrainer::opencl
#endif

namespace causallm {

/**
 * @brief RssAnon of this process, in KB. 0 if it cannot be read.
 */
inline size_t readRssAnonKb() {
#if defined(_WIN32)
  return 0;
#else
  std::FILE *f = std::fopen("/proc/self/status", "r");
  if (f == nullptr)
    return 0;
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "RssAnon:", 8) == 0) {
      kb = (size_t)std::strtoull(line + 8, nullptr, 10);
      break;
    }
  }
  std::fclose(f);
  return kb;
#endif
}

/**
 * @brief Bytes of dma-buf this process is holding open, from
 * /proc/self/fdinfo. This is the NPU term: the QNN path's weights and I/O
 * tensors are rpcmem allocations, each of which is a dma-buf fd whose fdinfo
 * carries a `size:` line. Exactly the sum an external observer takes over the
 * same files, so the two are the same quantity.
 */
inline size_t readDmabufBytes() {
#if defined(_WIN32)
  return 0;
#else
  DIR *d = opendir("/proc/self/fdinfo");
  if (d == nullptr)
    return 0;
  size_t total = 0;
  struct dirent *e;
  char path[64 + sizeof(((struct dirent *)0)->d_name)];
  char line[256];
  while ((e = readdir(d)) != nullptr) {
    if (e->d_name[0] < '0' || e->d_name[0] > '9')
      continue;
    std::snprintf(path, sizeof(path), "/proc/self/fdinfo/%s", e->d_name);
    std::FILE *f = std::fopen(path, "r");
    if (f == nullptr)
      continue;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      if (std::strncmp(line, "size:", 5) == 0) {
        total += (size_t)std::strtoull(line + 5, nullptr, 10);
        break;
      }
    }
    std::fclose(f);
  }
  closedir(d);
  return total;
#endif
}

/**
 * @brief Bytes this process holds on the GPU, from nntrainer's ledger.
 */
inline size_t readGpuBytes() {
#if defined(ENABLE_OPENCL)
  return nntrainer::opencl::clMemAcctLiveBytes();
#else
  return 0;
#endif
}

/**
 * @brief Peak of RssAnon + accelerator over one run.
 *
 * @details start() resets, so the value belongs to the request that called it
 * and not to the worker's history. A thread samples rather than the token
 * loop, because the peak of the GPU arm is inside load and inside the first
 * prefill chunk, neither of which has a per-token callback to hang off.
 */
class FootprintSampler {
public:
  static FootprintSampler &get() {
    static FootprintSampler s;
    return s;
  }

  /**
   * @brief Begin measuring, if nobody already is.
   *
   * @details Composable on purpose. A request that lazily loads the model
   * arms at the load and again at the generation; the second call is a no-op,
   * so the load's transient stays inside the request's peak instead of being
   * thrown away by a reset in between. A request that finds the model already
   * loaded arms only at the generation, and gets a peak that starts from zero
   * there. Either way the number belongs to one request.
   */
  void arm() {
    if (th_.joinable())
      return;
    {
      std::lock_guard<std::mutex> lk(m_);
      peak_kb_ = 0;
      peak_anon_kb_ = peak_gpu_kb_ = peak_dma_kb_ = 0;
      stop_ = false;
    }
    sample(); // one synchronous sample, so a run shorter than a tick reports
    th_ = std::thread([this] { loop(); });
  }

  /**
   * @brief Stop measuring and return the peak in KB.
   *
   * @details Does not clear: the next arm() does that. So an inner caller
   * (the model's own run) and an outer one (the API's request boundary) both
   * read the same number, in either order, and calling it twice is harmless.
   */
  size_t finish() {
    if (th_.joinable()) {
      {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
      }
      cv_.notify_all();
      th_.join();
      sample();
    }
    std::lock_guard<std::mutex> lk(m_);
    return peak_kb_;
  }

  size_t peakKb() {
    std::lock_guard<std::mutex> lk(m_);
    return peak_kb_;
  }

  /** The three terms as they stood at the peak, for the run summary. */
  void peakTerms(size_t &anon_kb, size_t &gpu_kb, size_t &dma_kb) {
    std::lock_guard<std::mutex> lk(m_);
    anon_kb = peak_anon_kb_;
    gpu_kb = peak_gpu_kb_;
    dma_kb = peak_dma_kb_;
  }

  /** Read the three terms once and fold them into the peak. */
  void sample() {
    const size_t anon_kb = readRssAnonKb();
    if (anon_kb == 0)
      return; // no /proc: leave the peak at 0 and let the caller fall back
    const size_t gpu_kb = readGpuBytes() / 1024;

    /** Every tick, not every third one. The NPU arm's peak is a transient
     *  inside the QNN graph preparation that a 450 ms cadence walks straight
     *  past (measured: 2 659 MiB reported against 2 860 MiB observed from
     *  outside, on a peak that lasts well under half a second). A scan is one
     *  open()/read()/close() per open fd, a few dozen of them. */
    dma_kb_ = readDmabufBytes() / 1024;

    /** One run uses one accelerator. Adding the two would count a mirror
     *  twice; the larger is the one that is really there. */
    const size_t accel_kb = gpu_kb > dma_kb_ ? gpu_kb : dma_kb_;

    std::lock_guard<std::mutex> lk(m_);
    const size_t now_kb = anon_kb + accel_kb;
    if (now_kb > peak_kb_) {
      peak_kb_ = now_kb;
      peak_anon_kb_ = anon_kb;
      peak_gpu_kb_ = gpu_kb;
      peak_dma_kb_ = dma_kb_;
    }
  }

private:
  FootprintSampler() {
    const char *e = std::getenv("NNTR_FOOTPRINT_MS");
    if (e != nullptr && e[0] != 0) {
      const long v = std::strtol(e, nullptr, 10);
      if (v >= 10 && v <= 5000)
        period_ms_ = (int)v;
    }
  }
  ~FootprintSampler() { finish(); }

  void loop() {
    for (;;) {
      std::unique_lock<std::mutex> lk(m_);
      if (cv_.wait_for(lk, std::chrono::milliseconds(period_ms_),
                       [this] { return stop_; }))
        return;
      lk.unlock();
      sample();
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::thread th_;
  bool stop_ = true;
  size_t peak_kb_ = 0;
  size_t peak_anon_kb_ = 0;
  size_t peak_gpu_kb_ = 0;
  size_t peak_dma_kb_ = 0;
  /** Touched by sample() only, and sample() is never concurrent: arm()
   *  takes its synchronous sample before the thread exists and finish() takes
   *  its last one after the join. */
  size_t dma_kb_ = 0;
  int period_ms_ = 100;
};

/**
 * @brief Arm the sampler for the duration of a scope, and read the peak out of
 * it on the way out.
 */
struct FootprintRun {
  FootprintRun() { FootprintSampler::get().arm(); }
  ~FootprintRun() { FootprintSampler::get().finish(); }
  FootprintRun(const FootprintRun &) = delete;
  FootprintRun &operator=(const FootprintRun &) = delete;
};

/**
 * @brief The number to report as this run's peak memory.
 *
 * @details The honest footprint when /proc gave one, and ru_maxrss when it did
 * not (Windows, or a kernel without RssAnon) so that the field is never empty.
 */
inline size_t resolvePeakMemoryKb(size_t honest_kb, size_t maxrss_kb) {
  return honest_kb > 0 ? honest_kb : maxrss_kb;
}

} // namespace causallm

#endif // __CAUSAL_LM_FOOTPRINT_SAMPLER_H__
