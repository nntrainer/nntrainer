// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 * @file pi05_cuda.cpp
 * @brief CUDA action expert: resident FP32 weights, cuBLAS and device
 * attention.
 */
#include "pi05_model.h"

#include <cuda_blas_manager.h>
#include <cuda_buffer.h>
#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace causallm {
namespace {
const char *expert_source = R"CU(
extern "C" {
__global__ void softmax(float *scores, int width) {
  __shared__ float red[256];
  int tid=threadIdx.x; scores+=(long)blockIdx.x*width;
  float mx=-3.402823466e38f;
  for(int j=tid;j<width;j+=256) mx=fmaxf(mx,scores[j]);
  red[tid]=mx; __syncthreads();
  for(int s=128;s;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}
  // All warps must read the maximum before reusing the reduction scratch.
  mx=red[0];__syncthreads();float sum=0;
  for(int j=tid;j<width;j+=256){scores[j]=expf(scores[j]-mx);sum+=scores[j];}
  red[tid]=sum;__syncthreads();
  for(int s=128;s;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
  sum=red[0];
  for(int j=tid;j<width;j+=256)scores[j]/=sum;
}
__global__ void bias(float *x, const float *b, int n, int d) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i]+=b[i%d];
}
__global__ void adarms(const float *x, const float *mod,
                       float *out, int d, float eps) {
  __shared__ float red[256];
  int r=blockIdx.x, tid=threadIdx.x;
  float sum=0;
  for(int i=tid;i<d;i+=256) sum+=x[r*d+i]*x[r*d+i];
  red[tid]=sum; __syncthreads();
  for(int s=128;s;s>>=1) { if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
  float inv=rsqrtf(red[0]/d+eps);
  for(int i=tid;i<d;i+=256)
    out[r*d+i]=x[r*d+i]*inv*(1+mod[i])+mod[d+i];
}
__global__ void residual(float *x, const float *branch, const float *mod,
                          int n, int d) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i]+=branch[i]*mod[2*d+i%d];
}
__global__ void geglu(const float *g, const float *u, float *out, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { float v=g[i]; out[i]=.5f*v*(1+tanhf(.7978845608028654f*
    (v+.044715f*v*v*v)))*u[i]; }
}
__global__ void rope(float *q, int rows, int heads, int hd, int pos,
                      float theta) {
  int i=blockIdx.x*blockDim.x+threadIdx.x, half=hd/2;
  if(i>=rows*heads*half) return;
  int j=i%half, h=(i/half)%heads, r=i/(heads*half);
  int base=(r*heads+h)*hd;
  float a=(pos+r)*powf(theta,-2.0f*j/hd), c=cosf(a), s=sinf(a);
  float x=q[base+j], y=q[base+half+j];
  q[base+j]=x*c-y*s; q[base+half+j]=y*c+x*s;
}
__global__ void append(const float *k, const float *v, float *kc, float *vc,
                        int n, int offset) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { kc[offset+i]=k[i]; vc[offset+i]=v[i]; }
}

}
)CU";
} // namespace

/** Own device weights and reusable activation buffers for one model instance.
 */
class PI05CudaExpert {
  using Buffer = nntrainer::cuda::Buffer;
  PI05Model &m;
  std::unordered_map<const float *, Buffer> weights;
  std::vector<Buffer> keys, values, norm_mods;
  unsigned int modulation_steps_ = 0;
  const float *active_mod_ = nullptr;
  Buffer x, norm, q, k, v, att, proj, gate, up, act, noise, velocity, scores;

  static Buffer allocate(size_t n) {
    Buffer b(n * sizeof(float));
    if (!b.GetBuffer())
      throw std::runtime_error("pi05: CUDA expert allocation failed");
    return b;
  }
  static float *ptr(Buffer &b) { return static_cast<float *>(b.GetBuffer()); }
  const float *weight(const std::vector<float> &w) {
    auto it = weights.find(w.data());
    if (it == weights.end()) {
      auto b = allocate(w.size());
      if (!b.WriteData(w.data()))
        throw std::runtime_error("pi05: CUDA expert weight copy failed");
      it = weights.emplace(w.data(), std::move(b)).first;
    }
    return static_cast<const float *>(it->second.GetBuffer());
  }
  template <typename... Args>
  void launch(const char *name, int blocks, size_t shared, Args... args) {
    auto kernel =
      nntrainer::CudaContext::Global().registerCudaKernel(expert_source, name);
    if (!kernel)
      throw std::runtime_error(std::string("pi05: CUDA kernel unavailable: ") +
                               name);
    unsigned int i = 0;
    (kernel->SetKernelArguments(i++, &args, sizeof(args)), ...);
    const int grid[3] = {blocks, 1, 1}, block[3] = {256, 1, 1};
    if (!nntrainer::cuda::StreamManager::Global().DispatchCommand(
          *kernel, grid, block, shared))
      throw std::runtime_error(std::string("pi05: CUDA launch failed: ") +
                               name);
  }
  void gemm(Buffer &input, const std::vector<float> &w, Buffer &output,
            int rows, int in, int out) {
    if (!nntrainer::cuda::BlasManager::Global().sgemmRowMajor(
          rows, out, in, ptr(input), weight(w), ptr(output)))
      throw std::runtime_error("pi05: CUDA expert GEMM failed");
  }
  void normalize(size_t index) {
    const int d = m.EXPERT_DIM;
    active_mod_ = ptr(norm_mods.at(index)) + m.per_step_v_t_.size() * 3 * d;
    launch("adarms", m.ACTION_HORIZON, 0, ptr(x), active_mod_, ptr(norm), d,
           m.RMS_EPS);
  }
  void addResidual() {
    const int n = m.ACTION_HORIZON * m.EXPERT_DIM;
    launch("residual", (n + 255) / 256, 0, ptr(x), ptr(proj), active_mod_, n,
           static_cast<int>(m.EXPERT_DIM));
  }

public:
  explicit PI05CudaExpert(PI05Model &model) : m(model) {
    const size_t T = m.ACTION_HORIZON, d = m.EXPERT_DIM;
    x = allocate(T * d);
    norm = allocate(T * d);
    proj = allocate(T * d);
    q = allocate(T * m.EXPERT_HEADS * m.EXPERT_HEAD_DIM);
    k = allocate(T * m.kv_width_);
    v = allocate(T * m.kv_width_);
    att = allocate(T * m.EXPERT_HEADS * m.EXPERT_HEAD_DIM);
    gate = allocate(T * m.EXPERT_MLP_DIM);
    up = allocate(T * m.EXPERT_MLP_DIM);
    act = allocate(T * m.EXPERT_MLP_DIM);
    noise = allocate(T * m.ACTION_DIM);
    velocity = allocate(T * m.ACTION_DIM);
  }
  void loadWeights() {
    for (const auto &layer : m.layers_)
      for (const auto *matrix : {&layer.qw, &layer.kw, &layer.vw, &layer.ow,
                                 &layer.gw, &layer.uw, &layer.dw})
        weight(*matrix);
    for (const auto *matrix :
         {&m.act_in_w, &m.act_in_b, &m.act_out_w, &m.act_out_b})
      weight(*matrix);
    // The module contains all expert kernels; compiling it needs no
    // observation.
    if (!nntrainer::CudaContext::Global().registerCudaKernel(expert_source,
                                                             "bias"))
      throw std::runtime_error("pi05: CUDA expert preparation failed");
  }
  void setPrefix() {
    if (modulation_steps_ != m.modulation_steps_) {
      norm_mods.clear();
      for (const auto &modulation : m.modulation_cache_) {
        auto buffer = allocate(modulation.size());
        if (!buffer.WriteData(modulation.data()))
          throw std::runtime_error("pi05: CUDA modulation copy failed");
        norm_mods.push_back(std::move(buffer));
      }
      modulation_steps_ = m.modulation_steps_;
    }
    keys.clear();
    values.clear();
    scores = allocate(m.EXPERT_HEADS * m.ACTION_HORIZON *
                      (m.prefix_len_ + m.ACTION_HORIZON));
    for (size_t i = 0; i < m.layers_.size(); ++i) {
      auto kb = allocate(m.k_cache_[i].size()),
           vb = allocate(m.v_cache_[i].size());
      if (!kb.WriteData(m.k_cache_[i].data()) ||
          !vb.WriteData(m.v_cache_[i].data()))
        throw std::runtime_error("pi05: CUDA prefix copy failed");
      keys.push_back(std::move(kb));
      values.push_back(std::move(vb));
    }
  }
  std::vector<float> run(const std::vector<float> &input,
                         const std::vector<float> &conditioning) {
    const int T = m.ACTION_HORIZON, d = m.EXPERT_DIM, hd = m.EXPERT_HEAD_DIM;
    const int H = m.EXPERT_HEADS, KVH = m.EXPERT_KV_HEADS, kw = m.kv_width_;
    const int total = m.prefix_len_ + T, mlp = m.EXPERT_MLP_DIM;
    (void)conditioning;
    if (!noise.WriteData(input.data()))
      throw std::runtime_error("pi05: CUDA expert input copy failed");
    gemm(noise, m.act_in_w, x, T, m.ACTION_DIM, d);
    launch("bias", (T * d + 255) / 256, 0, ptr(x), weight(m.act_in_b), T * d,
           d);
    for (size_t li = 0; li < m.layers_.size(); ++li) {
      const auto &l = m.layers_[li];
      normalize(2 * li);
      if (li == 0 && m.per_step_v_t_.empty() &&
          std::getenv("NNTR_PI05_DIAGNOSTICS")) {
        nntrainer::cuda::StreamManager::Global().finish();
        std::vector<float> tmp(T * d);
        norm.ReadData(tmp.data());
        PI05Model::dumpDiagnostic("l0_attn_in", tmp);
      }
      gemm(norm, l.qw, q, T, d, H * hd);
      gemm(norm, l.kw, k, T, d, kw);
      gemm(norm, l.vw, v, T, d, kw);
      launch("rope", (T * H * hd / 2 + 255) / 256, 0, ptr(q), T, H, hd,
             static_cast<int>(m.prefix_len_), m.ROPE_THETA);
      launch("rope", (T * KVH * hd / 2 + 255) / 256, 0, ptr(k), T, KVH, hd,
             static_cast<int>(m.prefix_len_), m.ROPE_THETA);
      launch("append", (T * kw + 255) / 256, 0, ptr(k), ptr(v), ptr(keys[li]),
             ptr(values[li]), T * kw, static_cast<int>(m.prefix_len_ * kw));
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      const float zero = 0, one = 1;
      auto handle = nntrainer::cuda::BlasManager::Global().handle();
      for (int kh = 0; kh < KVH; ++kh) {
        const int groups = H / KVH;
        if (cublasSgemmStridedBatched(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, total, T, hd, &scale,
              ptr(keys[li]) + kh * hd, kw, 0, ptr(q) + kh * groups * hd, H * hd,
              hd, &zero,
              ptr(scores) + static_cast<size_t>(kh * groups) * T * total, total,
              static_cast<long long>(T) * total,
              groups) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("pi05: expert QK GEMM failed");
      }
      launch("softmax", T * H, 0, ptr(scores), total);
      for (int kh = 0; kh < KVH; ++kh) {
        const int groups = H / KVH;
        if (cublasSgemmStridedBatched(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, hd, T, total, &one,
              ptr(values[li]) + kh * hd, kw, 0,
              ptr(scores) + static_cast<size_t>(kh * groups) * T * total, total,
              static_cast<long long>(T) * total, &zero,
              ptr(att) + kh * groups * hd, H * hd, hd,
              groups) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("pi05: expert PV GEMM failed");
      }
      gemm(att, l.ow, proj, T, H * hd, d);
      addResidual();
      if (li == 0 && m.per_step_v_t_.empty() &&
          std::getenv("NNTR_PI05_DIAGNOSTICS")) {
        nntrainer::cuda::StreamManager::Global().finish();
        std::vector<float> tmp(T * d);
        x.ReadData(tmp.data());
        PI05Model::dumpDiagnostic("l0_after_attn", tmp);
      }
      normalize(2 * li + 1);
      gemm(norm, l.gw, gate, T, d, mlp);
      gemm(norm, l.uw, up, T, d, mlp);
      launch("geglu", (T * mlp + 255) / 256, 0, ptr(gate), ptr(up), ptr(act),
             T * mlp);
      gemm(act, l.dw, proj, T, mlp, d);
      addResidual();
    }
    normalize(2 * m.layers_.size());
    gemm(norm, m.act_out_w, velocity, T, d, m.ACTION_DIM);
    const int count = T * m.ACTION_DIM;
    launch("bias", (count + 255) / 256, 0, ptr(velocity), weight(m.act_out_b),
           count, static_cast<int>(m.ACTION_DIM));
    nntrainer::cuda::StreamManager::Global().finish();
    std::vector<float> result(count);
    if (!velocity.ReadData(result.data()))
      throw std::runtime_error("pi05: CUDA expert output copy failed");
    return result;
  }
};

void PI05Model::loadCudaExpertWeights() {
  cuda_expert_ = std::make_shared<PI05CudaExpert>(*this);
  cuda_expert_->loadWeights();
}

void PI05Model::prepareCudaExpert() {
  if (!cuda_expert_)
    cuda_expert_ = std::make_shared<PI05CudaExpert>(*this);
  cuda_expert_->setPrefix();
}
std::vector<float> PI05Model::denoiseStepCuda(const std::vector<float> &x,
                                              const std::vector<float> &cond) {
  return cuda_expert_->run(x, cond);
}
} // namespace causallm
