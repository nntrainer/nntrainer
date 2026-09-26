// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_emb_gather.cpp
 * @date    19 Aug 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   On-GPU embedding-LUT gather+dequant -- see cuda_emb_gather.h.
 */

#include "cuda_emb_gather.h"

#include "cuda_fc_qs4cx.h" // [lmhead-tie-lut] VRAM LUT copy lookup

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace nntrainer::cuda {

// Gather kernels. eg_f2h is the house RN float->half (identical to ew_f2h in
// cuda_elementwise.cpp), so the fp16 store matches the host's
// static_cast<_FP16>(float) bit-for-bit. The dequant replays the host
// forEachPacked4BitValue arithmetic exactly: two's-complement signed nibble
// (low nibble first), fp32 (value * block_scale * layer_scale) with the same
// multiply order, scale index = row * nblocks + col / block_width. One thread
// per column; the row id is read from the device slot (tok[0]) so a captured
// graph gathers the CURRENT token's row on every replay.
static const char *EMB_GATHER_SRC = R"CU(
extern "C" {
__device__ __forceinline__ unsigned short eg_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
}
__device__ __forceinline__ float eg_dequant_s4(const unsigned char *lut,
                                               const float *scales, int row,
                                               int c, int out_dim, int nblocks,
                                               int block_width,
                                               float layer_scale) {
  unsigned char b = lut[(long long)row * (out_dim >> 1) + (c >> 1)];
  int nib = (c & 1) ? (b >> 4) : (b & 0x0f);
  int sv = (nib & 0x08) ? nib - 16 : nib;
  float s = scales[(long long)row * nblocks + c / block_width];
  return (float)sv * s * layer_scale;
}
__global__ void emb_gather_s4_f16(const unsigned char *lut,
                                  const float *scales, const int *tok,
                                  int out_dim, int nblocks, int block_width,
                                  int n_rows, float layer_scale,
                                  unsigned short *out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out_dim) return;
  int row = tok[0];
  if (row < 0 || row >= n_rows) return;
  out[c] = eg_f2h(
    eg_dequant_s4(lut, scales, row, c, out_dim, nblocks, block_width,
                  layer_scale));
}
// Batched page warm: one thread per id, touching the id's payload-row pages
// and scale row so the HMM faults are raised in ONE kernel (driver batches
// them) instead of one-at-a-time inside the decode gathers. The reads are
// dead (guarded impossible store) -- only the page mappings matter.
__global__ void emb_warm_rows(const unsigned char *lut, const float *scales,
                              const int *ids, int n, int row_bytes,
                              int nblocks, int n_rows,
                              unsigned long long *sink) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int row = ids[i];
  if (row < 0 || row >= n_rows) return;
  const unsigned char *r = lut + (long long)row * row_bytes;
  unsigned long long s = 0;
  for (int o = 0; o < row_bytes; o += 4096)
    s += r[o];
  s += r[row_bytes - 1];
  const float *sc = scales + (long long)row * nblocks;
  s += (unsigned long long)__float_as_int(sc[0]);
  s += (unsigned long long)__float_as_int(sc[nblocks - 1]);
  if (s == 0x1234567887654321ull)
    sink[0] = s; // never taken; keeps the reads alive
}
__global__ void emb_gather_s4_f32(const unsigned char *lut,
                                  const float *scales, const int *tok,
                                  int out_dim, int nblocks, int block_width,
                                  int n_rows, float layer_scale, float *out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out_dim) return;
  int row = tok[0];
  if (row < 0 || row >= n_rows) return;
  out[c] = eg_dequant_s4(lut, scales, row, c, out_dim, nblocks, block_width,
                         layer_scale);
}
}
)CU";

namespace {

struct GatherLut {
  const uint8_t *payload = nullptr;
  const float *scales = nullptr;
  unsigned n_rows = 0;
  unsigned out_dim = 0;
  unsigned nblocks = 0;
  size_t row_bytes = 0;
  size_t scale_row_bytes = 0;
  std::vector<bool> warm; ///< per-row: GPU pages prefetched already
};

std::vector<GatherLut> g_luts;        // small (2 for the folded demo)
int *g_tok_dev = nullptr;             // device id slot the graph bakes in
int *g_tok_pin = nullptr;             // pinned host slot; feed = one store
cudaStream_t g_side_stream = nullptr; // page-warm prefetches (off critical)
bool g_graph_live = false;
unsigned g_epoch = 0;

// Prefill warm-batch state: a small pool of pinned id-staging buffers, each
// guarded by its own completion event so every chunk (x every LUT) can
// enqueue its batch without waiting on the previous one -- the single-buffer
// version silently warmed only the first chunk. Pool capped; if every buffer
// is still in flight the batch is skipped (decode-time notify prefetch stays
// as the safety net).
struct WarmBuf {
  int *pin = nullptr;
  unsigned cap = 0;
  cudaEvent_t evt = nullptr;
  bool inflight = false;
};
std::vector<WarmBuf> g_warm_bufs;
constexpr size_t WARM_POOL_CAP = 8;
unsigned long long *g_warm_sink = nullptr;

// Default-ON kill switch (NNTR_CUDA_EMB_GATHER=0 restores the host
// dequant+staging feed on every path).
bool gather_env_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CUDA_EMB_GATHER");
    return !(e && e[0] == '0');
  }();
  return on;
}

// HMM full pageable access: the GPU can dereference ANY host pointer,
// including the read-only file mmap the LUT payload lives in. Without it
// (Windows/WDDM, proprietary module, old kernels) registration fails and the
// host path is kept -- cudaHostRegister on a file-backed mapping is NOT a
// fallback (rejected by the driver, and it would pin the whole table).
bool hmm_pageable_access() {
  static const bool ok = []() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
      return false;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, dev) !=
        cudaSuccess)
      return false;
    return v == 1;
  }();
  return ok;
}

// Best-effort: map-on-demand hint + page warm. Failures are swallowed -- the
// gather kernels fault the pages themselves (slower first touch, same bytes).
void advise_accessed_by(const void *p, size_t bytes) {
  int dev = 0;
  cudaGetDevice(&dev);
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, loc);
#else
  cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, dev);
#endif
  cudaGetLastError();
}

void prefetch_range(const void *p, size_t bytes) {
  if (g_side_stream == nullptr)
    return;
  const uintptr_t page = 4096;
  uintptr_t b = reinterpret_cast<uintptr_t>(p) & ~(page - 1);
  size_t len =
    ((reinterpret_cast<uintptr_t>(p) + bytes + page - 1) & ~(page - 1)) - b;
  int dev = 0;
  cudaGetDevice(&dev);
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  cudaMemPrefetchAsync(reinterpret_cast<const void *>(b), len, loc, 0,
                       g_side_stream);
#else
  cudaMemPrefetchAsync(reinterpret_cast<const void *>(b), len, dev,
                       g_side_stream);
#endif
  cudaGetLastError();
}

// Warm one id's payload row + scale row on the side stream (once per id).
// Never under capture: cudaMemPrefetchAsync is not a capturable op.
void warm_token(GatherLut &lut, unsigned tok) {
  if (tok >= lut.n_rows || lut.warm[tok])
    return;
  if (StreamManager::Global().isCapturing())
    return;
  prefetch_range(lut.payload + (size_t)tok * lut.row_bytes, lut.row_bytes);
  prefetch_range(reinterpret_cast<const uint8_t *>(lut.scales) +
                   (size_t)tok * lut.scale_row_bytes,
                 lut.scale_row_bytes);
  lut.warm[tok] = true;
}

} // namespace

int emb_gather_register_lut(const void *payload, size_t payload_bytes,
                            const float *scales, size_t scale_count,
                            unsigned n_rows, unsigned out_dim,
                            unsigned nblocks) {
  if (!gather_env_on() || !hmm_pageable_access())
    return -1;
  if (payload == nullptr || scales == nullptr || n_rows == 0 || out_dim == 0 ||
      (out_dim % 2) != 0 || nblocks == 0 || (out_dim % nblocks) != 0)
    return -1;
  if (payload_bytes != (size_t)n_rows * (out_dim / 2) ||
      scale_count != (size_t)n_rows * nblocks)
    return -1;
  // No allocation / NVRTC module load may happen under capture.
  if (StreamManager::Global().isCapturing())
    return -1;

  for (size_t i = 0; i < g_luts.size(); ++i)
    if (g_luts[i].payload == payload)
      return (int)i;

  // Shared id slot + side stream (once).
  if (g_tok_dev == nullptr &&
      cudaMalloc((void **)&g_tok_dev, sizeof(int)) != cudaSuccess) {
    g_tok_dev = nullptr;
    cudaGetLastError();
    return -1;
  }
  if (g_tok_pin == nullptr &&
      cudaHostAlloc((void **)&g_tok_pin, sizeof(int), cudaHostAllocDefault) !=
        cudaSuccess) {
    g_tok_pin = nullptr;
    cudaGetLastError();
    return -1;
  }
  if (g_side_stream == nullptr &&
      cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
    g_side_stream = nullptr;
    cudaGetLastError(); // prefetch warm-ups degrade to in-kernel faults
  }

  // Compile/load the kernels NOW (module load under capture is illegal).
  auto k16 = CudaContext::Global().registerCudaKernel(EMB_GATHER_SRC,
                                                      "emb_gather_s4_f16");
  auto k32 = CudaContext::Global().registerCudaKernel(EMB_GATHER_SRC,
                                                      "emb_gather_s4_f32");
  auto kw =
    CudaContext::Global().registerCudaKernel(EMB_GATHER_SRC, "emb_warm_rows");
  if (!k16 || !k32 || !kw) {
    ml_loge("[CUDA] emb_gather: kernel registration failed");
    return -1;
  }

  GatherLut lut;
  lut.payload = static_cast<const uint8_t *>(payload);
  lut.scales = scales;
  lut.n_rows = n_rows;
  lut.out_dim = out_dim;
  lut.nblocks = nblocks;
  lut.row_bytes = out_dim / 2;
  lut.scale_row_bytes = (size_t)nblocks * sizeof(float);
  // [lmhead-tie-lut] If the app uploaded a VRAM copy of this very payload for
  // the tied lm_head GEMV, gather from it instead of HMM zero-copy: same
  // bytes (verbatim device copy of payload + fp32 scales), but every row is
  // resident, so the per-new-token HMM mapping fault (and the side-stream
  // prefetch machinery that hides it) disappears for this table. Marking the
  // whole warm bitmap keeps warm_token()/emb_gather_warm_ids() no-ops --
  // which also keeps cudaMemPrefetchAsync away from a non-managed pointer.
  const void *dev_payload = nullptr;
  const float *dev_scales = nullptr;
  const bool vram_copy =
    cuda_fc_lmhead_tie_lut_device_copy(payload, &dev_payload, &dev_scales);
  if (vram_copy) {
    lut.payload = static_cast<const uint8_t *>(dev_payload);
    lut.scales = dev_scales;
    lut.warm.assign(n_rows, true);
  } else {
    lut.warm.assign(n_rows, false);
    advise_accessed_by(payload, payload_bytes);
    advise_accessed_by(scales, scale_count * sizeof(float));
  }
  g_luts.push_back(std::move(lut));
  ml_logi("[CUDA] emb_gather: LUT %zu registered (rows=%u out_dim=%u "
          "blocks=%u, %s)",
          g_luts.size() - 1, n_rows, out_dim, nblocks,
          vram_copy ? "tied VRAM copy" : "zero-copy HMM");
  return (int)(g_luts.size() - 1);
}

void emb_gather_warm_ids(int handle, const float *ids_host, unsigned n) {
  // Opt-in (measured on the 29K demo): the prompt vocabulary covered only
  // ~10% of the decode-novel ids while the side-stream fault storm cost the
  // prefill ~100 ms, so warming the prompt rows is a net loss there. Kept as
  // a lever for workloads whose decode vocabulary DOES track the prompt.
  static const bool warm_on = []() {
    const char *e = std::getenv("NNTR_CUDA_EMB_GATHER_PREFILL_WARM");
    return e && e[0] == '1';
  }();
  if (!warm_on || handle < 0 || (size_t)handle >= g_luts.size() ||
      ids_host == nullptr || n == 0 || g_side_stream == nullptr)
    return;
  auto &sm = StreamManager::Global();
  if (sm.isCapturing())
    return;
  if (g_warm_sink == nullptr &&
      cudaMalloc((void **)&g_warm_sink, sizeof(unsigned long long)) !=
        cudaSuccess) {
    g_warm_sink = nullptr;
    cudaGetLastError();
    return;
  }
  // Claim a free (completed) staging buffer, growing the pool up to its cap.
  WarmBuf *buf = nullptr;
  for (auto &b : g_warm_bufs) {
    if (b.inflight) {
      if (cudaEventQuery(b.evt) != cudaSuccess) {
        cudaGetLastError();
        continue;
      }
      b.inflight = false;
    }
    buf = &b;
    break;
  }
  if (buf == nullptr) {
    if (g_warm_bufs.size() >= WARM_POOL_CAP)
      return; // every buffer busy: skip; notify prefetch covers the rest
    g_warm_bufs.emplace_back();
    buf = &g_warm_bufs.back();
    if (cudaEventCreateWithFlags(&buf->evt, cudaEventDisableTiming) !=
        cudaSuccess) {
      buf->evt = nullptr;
      g_warm_bufs.pop_back();
      cudaGetLastError();
      return;
    }
  }
  if (n > buf->cap) {
    if (buf->pin)
      cudaFreeHost(buf->pin);
    if (cudaHostAlloc((void **)&buf->pin, n * sizeof(int),
                      cudaHostAllocDefault) != cudaSuccess) {
      buf->pin = nullptr;
      buf->cap = 0;
      cudaGetLastError();
      return;
    }
    buf->cap = n;
  }

  GatherLut &lut = g_luts[(size_t)handle];
  unsigned fresh = 0;
  for (unsigned i = 0; i < n; ++i) {
    const unsigned tok = (unsigned)ids_host[i];
    if (tok >= lut.n_rows || lut.warm[tok])
      continue;
    lut.warm[tok] = true; // marked at submit; the kernel below faults it in
    buf->pin[fresh++] = (int)tok;
  }
  if (fresh == 0)
    return;

  auto kernel =
    CudaContext::Global().registerCudaKernel(EMB_GATHER_SRC, "emb_warm_rows");
  if (!kernel)
    return;
  // The kernel reads the pinned id list zero-copy; the event below guards the
  // single staging buffer against rewrite while the batch is in flight.
  const int *ids_dev = buf->pin;
  int ni = (int)fresh, row_bytes = (int)lut.row_bytes;
  int nblocks = (int)lut.nblocks, n_rows = (int)lut.n_rows;
  kernel->SetKernelArguments(0, &lut.payload, sizeof(lut.payload));
  kernel->SetKernelArguments(1, &lut.scales, sizeof(lut.scales));
  kernel->SetKernelArguments(2, &ids_dev, sizeof(ids_dev));
  kernel->SetKernelArguments(3, &ni, sizeof(ni));
  kernel->SetKernelArguments(4, &row_bytes, sizeof(row_bytes));
  kernel->SetKernelArguments(5, &nblocks, sizeof(nblocks));
  kernel->SetKernelArguments(6, &n_rows, sizeof(n_rows));
  kernel->SetKernelArguments(7, &g_warm_sink, sizeof(g_warm_sink));
  const int block[3] = {128, 1, 1};
  const int grid[3] = {(int)((fresh + 127) / 128), 1, 1};
  auto params = kernel->getKernelParams();
  if (cuLaunchKernel(kernel->GetFunction(), grid[0], 1, 1, block[0], 1, 1, 0,
                     reinterpret_cast<CUstream>(g_side_stream), params.data(),
                     nullptr) != CUDA_SUCCESS) {
    cudaGetLastError();
    return;
  }
  if (cudaEventRecord(buf->evt, g_side_stream) == cudaSuccess)
    buf->inflight = true;
  else
    cudaGetLastError();
}

void emb_gather_set_token(int handle, int tok) {
  if (handle < 0 || (size_t)handle >= g_luts.size() || g_tok_pin == nullptr)
    return;
  *g_tok_pin = tok; // consumed by the captured (or eager) 4-byte H2D
  if (tok >= 0)
    warm_token(g_luts[(size_t)handle], (unsigned)tok); // host-sampled ids
}

void emb_gather_notify_token(unsigned tok) {
  for (auto &lut : g_luts)
    warm_token(lut, tok);
}

bool emb_gather_dispatch_s4(int handle, float layer_scale, void *out,
                            bool fp16_out) {
  if (handle < 0 || (size_t)handle >= g_luts.size() || out == nullptr ||
      g_tok_dev == nullptr || g_tok_pin == nullptr)
    return false;
  const GatherLut &lut = g_luts[(size_t)handle];

  auto kernel = CudaContext::Global().registerCudaKernel(
    EMB_GATHER_SRC, fp16_out ? "emb_gather_s4_f16" : "emb_gather_s4_f32");
  if (!kernel)
    return false;

  auto &sm = StreamManager::Global();
  // Token id: pinned slot -> device slot. Captured into the decode graph
  // when recording, so every replay re-reads the slot the feed pass wrote.
  if (!sm.EnqueueWriteBuffer(g_tok_dev, sizeof(int), g_tok_pin, true))
    return false;

  int out_dim = (int)lut.out_dim, nblocks = (int)lut.nblocks;
  int block_width = (int)(lut.out_dim / lut.nblocks);
  int n_rows = (int)lut.n_rows;
  kernel->SetKernelArguments(0, &lut.payload, sizeof(lut.payload));
  kernel->SetKernelArguments(1, &lut.scales, sizeof(lut.scales));
  kernel->SetKernelArguments(2, &g_tok_dev, sizeof(g_tok_dev));
  kernel->SetKernelArguments(3, &out_dim, sizeof(out_dim));
  kernel->SetKernelArguments(4, &nblocks, sizeof(nblocks));
  kernel->SetKernelArguments(5, &block_width, sizeof(block_width));
  kernel->SetKernelArguments(6, &n_rows, sizeof(n_rows));
  kernel->SetKernelArguments(7, &layer_scale, sizeof(layer_scale));
  kernel->SetKernelArguments(8, &out, sizeof(out));
  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((lut.out_dim + 255) / 256), 1, 1};
  if (!sm.DispatchCommand(*kernel, grid, block))
    return false;
  sm.maybeFinish();
  return true;
}

void emb_gather_set_graph_live(bool live) {
  if (g_graph_live && !live)
    ++g_epoch; // the graph (and any gather nodes captured in it) is gone
  g_graph_live = live;
}

bool emb_gather_graph_live() { return g_graph_live; }

unsigned emb_gather_epoch() { return g_epoch; }

} // namespace nntrainer::cuda
