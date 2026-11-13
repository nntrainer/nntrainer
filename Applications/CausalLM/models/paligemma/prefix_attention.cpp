// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 * @file prefix_attention.cpp
 * @brief FP32 noncausal attention for vision and exported pi0.5 prefix K/V.
 */
#include "prefix_attention.h"
#include <algorithm>
#include <cmath>
#include <cpu_backend.h>
#include <cstring>
#include <fp16.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <layer_node.h>
#include <llm_util.hpp>
#include <stdexcept>
#include <thread_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_blas_manager.h>
#include <cuda_buffer.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>
#endif
namespace causallm {
namespace {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
const char *prefix_convert_source = R"CU(
extern "C" {
__device__ float load_value(const void *p,int i,int half) {
  if(!half)return ((const float*)p)[i];
  unsigned short v=((const unsigned short*)p)[i];
  float out; asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(v)); return out;
}
__global__ void convert_rope(const void *in,float *out,int n,int heads,int hd,
                              int half,float theta) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  float value=load_value(in,i,half);
  if(theta>0) {
    int d=i%hd,j=d%(hd/2),r=i/(heads*hd);
    int partner=d<hd/2?i+hd/2:i-hd/2;
    float angle=r/powf(theta,2.0f*j/hd);
    float c=cosf(angle),s=sinf(angle);
    float other=load_value(in,partner,half);
    value=d<hd/2?value*c-other*s:value*c+other*s;
  }
  out[i]=value;
}
__global__ void convert_output(const float *in,void *out,int n,int half) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  if(!half)((float*)out)[i]=in[i];
  else { unsigned short v; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(v) : "f"(in[i]));
    ((unsigned short*)out)[i]=v; }
}
}
)CU";

template <typename... Args>
void prefixLaunch(const char *name, unsigned int n, Args... args) {
  auto kernel = nntrainer::CudaContext::Global().registerCudaKernel(
    prefix_convert_source, name);
  if (!kernel)
    throw std::runtime_error("prefix conversion kernel unavailable");
  unsigned int i = 0;
  (kernel->SetKernelArguments(i++, &args, sizeof(args)), ...);
  const int grid[3] = {static_cast<int>((n + 255) / 256), 1, 1};
  const int block[3] = {256, 1, 1};
  if (!nntrainer::cuda::StreamManager::Global().DispatchCommand(*kernel, grid,
                                                                block))
    throw std::runtime_error("prefix conversion launch failed");
}

void prefixCuda(const float *qp, const float *kp, const float *vp, float *op,
                unsigned int heads, unsigned int kv_heads, unsigned int dim,
                unsigned int rows, bool interleaved) {
  using nntrainer::cuda::Buffer;
  const unsigned int q_ld = interleaved ? heads * dim : dim;
  const unsigned int kv_ld = interleaved ? kv_heads * dim : dim;
  const size_t q_head_stride = interleaved ? dim : rows * dim;
  const size_t kv_head_stride = interleaved ? dim : rows * dim;
  Buffer scores(static_cast<size_t>(heads) * rows * rows * sizeof(float));
  if (!scores.GetBuffer())
    throw std::runtime_error("FP32 prefix CUDA allocation failed");
  auto handle = nntrainer::cuda::BlasManager::Global().handle();
  auto *sp = static_cast<float *>(scores.GetBuffer());
  const float scale = 1 / std::sqrt(static_cast<float>(dim)), zero = 0, one = 1;
  if (kv_heads == heads || kv_heads == 1) {
    if (cublasSgemmStridedBatched(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, rows, dim, &scale, kp, kv_ld,
          kv_heads == 1 ? 0 : kv_head_stride, qp, q_ld, q_head_stride, &zero,
          sp, rows, static_cast<long long>(rows) * rows,
          heads) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("prefix batched QK GEMM failed");
  } else
    for (unsigned int h = 0; h < heads; ++h) {
      const unsigned int kh = h / (heads / kv_heads);
      if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, rows, dim, &scale,
                      kp + static_cast<size_t>(kh) * kv_head_stride, kv_ld,
                      qp + static_cast<size_t>(h) * q_head_stride, q_ld, &zero,
                      sp + static_cast<size_t>(h) * rows * rows,
                      rows) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("prefix QK GEMM failed");
    }
  static const char *softmax_source = R"CU(
extern "C" __global__ void prefix_softmax(float *x,int width) {
  __shared__ float reduce[256];
  int row=blockIdx.x,t=threadIdx.x; x+=(long)row*width;
  float mx=-3.402823466e38f;
  for(int j=t;j<width;j+=256) mx=fmaxf(mx,x[j]);
  reduce[t]=mx;__syncthreads();
  for(int s=128;s;s>>=1){if(t<s)reduce[t]=fmaxf(reduce[t],reduce[t+s]);__syncthreads();}
  // All warps must read the maximum before reusing the reduction scratch.
  mx=reduce[0];__syncthreads();float sum=0;
  for(int j=t;j<width;j+=256){x[j]=expf(x[j]-mx);sum+=x[j];}
  reduce[t]=sum;__syncthreads();
  for(int s=128;s;s>>=1){if(t<s)reduce[t]+=reduce[t+s];__syncthreads();}
  sum=reduce[0];
  for(int j=t;j<width;j+=256)x[j]/=sum;
}
)CU";
  auto kernel = nntrainer::CudaContext::Global().registerCudaKernel(
    softmax_source, "prefix_softmax");
  if (!kernel)
    throw std::runtime_error("prefix softmax kernel unavailable");
  const int width = rows;
  kernel->SetKernelArguments(0, &sp, sizeof(sp));
  kernel->SetKernelArguments(1, &width, sizeof(width));
  const int grid[3] = {static_cast<int>(heads * rows), 1, 1},
            block[3] = {256, 1, 1};
  if (!nntrainer::cuda::StreamManager::Global().DispatchCommand(*kernel, grid,
                                                                block))
    throw std::runtime_error("prefix softmax launch failed");
  if (kv_heads == heads || kv_heads == 1) {
    if (cublasSgemmStridedBatched(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, rows, rows, &one, vp, kv_ld,
          kv_heads == 1 ? 0 : kv_head_stride, sp, rows,
          static_cast<long long>(rows) * rows, &zero, op, q_ld, q_head_stride,
          heads) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("prefix batched PV GEMM failed");
  } else
    for (unsigned int h = 0; h < heads; ++h) {
      const unsigned int kh = h / (heads / kv_heads);
      if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, rows, rows, &one,
                      vp + static_cast<size_t>(kh) * kv_head_stride, kv_ld,
                      sp + static_cast<size_t>(h) * rows * rows, rows, &zero,
                      op + static_cast<size_t>(h) * q_head_stride,
                      q_ld) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("prefix PV GEMM failed");
    }
  nntrainer::cuda::StreamManager::Global().finish();
}
#endif
class PrefixAttention : public nntrainer::LayerImpl {
  unsigned int heads_, kv_heads_, dim_;
  float theta_;
  std::shared_ptr<PrefixKV> cache_;

public:
  PrefixAttention(unsigned int heads, unsigned int kv_heads, unsigned int dim,
                  float theta, std::shared_ptr<PrefixKV> cache) :
    heads_(heads),
    kv_heads_(kv_heads),
    dim_(dim),
    theta_(theta),
    cache_(std::move(cache)) {}
  const std::string getType() const override { return "prefix_attention"; }
  void finalize(nntrainer::InitLayerContext &ctx) override {
    if (ctx.getNumInputs() != 3 || ctx.getInputDimensions()[0].batch() != 1)
      throw std::invalid_argument(
        "prefix_attention requires three batch-one inputs");
    auto out = ctx.getInputDimensions()[0];
    out.setTensorType({ctx.getFormat(), ctx.getActivationDataType()});
    ctx.setOutputDimensions({out});
  }
  void forwarding(nntrainer::RunLayerContext &ctx, bool training) override {
    incremental_forwarding(ctx, 0, ctx.getInput(0).height(), training);
  }
  void incremental_forwarding(nntrainer::RunLayerContext &ctx,
                              unsigned int from, unsigned int to,
                              bool) override {
    if (from || to == 0 || to > ctx.getInput(0).height())
      throw std::invalid_argument(
        "prefix_attention expects a complete prefix at position zero");
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    const bool gpu = causallm_engine() == "cuda";
    if (gpu)
      nntrainer::cuda::StreamManager::Global().finish();
#endif
    const unsigned int rows = to;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // SigLIP Q/K/V and the following projection already live on the device.
    // cuBLAS leading dimensions consume their interleaved layout directly,
    // avoiding a managed-memory round trip and host transpose at every layer.
    if (gpu && theta_ <= 0 && !cache_ &&
        ctx.getInput(0).getDataType() == nntrainer::Tdatatype::FP32 &&
        ctx.getInput(1).getDataType() == nntrainer::Tdatatype::FP32 &&
        ctx.getInput(2).getDataType() == nntrainer::Tdatatype::FP32 &&
        ctx.getOutput(0).getDataType() == nntrainer::Tdatatype::FP32) {
      const auto *q = ctx.getInput(0).getData<float>();
      const auto *k = ctx.getInput(1).getData<float>();
      const auto *v = ctx.getInput(2).getData<float>();
      auto *out = ctx.getOutput(0).getData<float>();
      if (nntrainer::cuda::dev_accessible(q) &&
          nntrainer::cuda::dev_accessible(k) &&
          nntrainer::cuda::dev_accessible(v) &&
          nntrainer::cuda::dev_accessible(out)) {
        prefixCuda(q, k, v, out, heads_, kv_heads_, dim_, rows, true);
        return;
      }
    }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    if (gpu && nntrainer::cuda::dev_accessible(ctx.getInput(0).getData()) &&
        nntrainer::cuda::dev_accessible(ctx.getInput(1).getData()) &&
        nntrainer::cuda::dev_accessible(ctx.getInput(2).getData()) &&
        nntrainer::cuda::dev_accessible(ctx.getOutput(0).getData())) {
      const unsigned int nq = rows * heads_ * dim_;
      const unsigned int nk = rows * kv_heads_ * dim_;
      nntrainer::cuda::Buffer qb(nq * sizeof(float)), kb(nk * sizeof(float)),
        vb(nk * sizeof(float)), ob(nq * sizeof(float));
      if (!qb.GetBuffer() || !kb.GetBuffer() || !vb.GetBuffer() ||
          !ob.GetBuffer())
        throw std::runtime_error("prefix conversion allocation failed");
      auto *q = static_cast<float *>(qb.GetBuffer());
      auto *k = static_cast<float *>(kb.GetBuffer());
      auto *v = static_cast<float *>(vb.GetBuffer());
      auto *o = static_cast<float *>(ob.GetBuffer());
      auto convert = [&](unsigned int index, float *dest, unsigned int n,
                         unsigned int heads, float theta) {
        const int half =
          ctx.getInput(index).getDataType() == nntrainer::Tdatatype::FP16;
        prefixLaunch("convert_rope", n, ctx.getInput(index).getData(), dest,
                     static_cast<int>(n), static_cast<int>(heads),
                     static_cast<int>(dim_), half, theta);
      };
      convert(0, q, nq, heads_, theta_);
      convert(1, k, nk, kv_heads_, theta_);
      convert(2, v, nk, kv_heads_, 0.0f);
      prefixCuda(q, k, v, o, heads_, kv_heads_, dim_, rows, true);
      if (cache_) {
        cache_->key.resize(nk);
        cache_->value.resize(nk);
        if (!kb.ReadData(cache_->key.data()) ||
            !vb.ReadData(cache_->value.data()))
          throw std::runtime_error("prefix cache readback failed");
      }
      const int half =
        ctx.getOutput(0).getDataType() == nntrainer::Tdatatype::FP16;
      prefixLaunch("convert_output", nq, o, ctx.getOutput(0).getData(),
                   static_cast<int>(nq), half);
      nntrainer::cuda::StreamManager::Global().finish();
      return;
    }
#endif
    auto read = [rows](const nntrainer::Tensor &t) {
      const size_t n = static_cast<size_t>(rows) * t.width();
      std::vector<float> out(n);
      if (t.getDataType() == nntrainer::Tdatatype::FP32)
        std::copy(t.getData<float>(), t.getData<float>() + n, out.begin());
      else {
        const auto *bits = reinterpret_cast<const uint16_t *>(t.getData());
        std::transform(bits, bits + n, out.begin(),
                       nntrainer::compute_fp16_to_fp32);
      }
      return out;
    };
    auto q = read(ctx.getInput(0)), k = read(ctx.getInput(1)),
         v = read(ctx.getInput(2));
    auto rope = [this, rows](std::vector<float> &x, unsigned int heads) {
      if (theta_ <= 0)
        return;
      const unsigned int half = dim_ / 2;
      for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int h = 0; h < heads; ++h)
          for (unsigned int j = 0; j < half; ++j) {
            const float angle = r / std::pow(theta_, 2.0f * j / dim_);
            const float c = std::cos(angle), s = std::sin(angle);
            float *p = x.data() + (static_cast<size_t>(r) * heads + h) * dim_;
            const float a = p[j], b = p[j + half];
            p[j] = a * c - b * s;
            p[j + half] = b * c + a * s;
          }
    };
    rope(q, heads_);
    rope(k, kv_heads_);
    if (cache_) {
      cache_->key = k;
      cache_->value = v;
    }
    auto headMajor = [rows, this](const std::vector<float> &x,
                                  unsigned int heads) {
      std::vector<float> out(x.size());
      for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int h = 0; h < heads; ++h)
          std::copy_n(x.data() + (static_cast<size_t>(r) * heads + h) * dim_,
                      dim_,
                      out.data() + (static_cast<size_t>(h) * rows + r) * dim_);
      return out;
    };
    q = headMajor(q, heads_);
    k = headMajor(k, kv_heads_);
    v = headMajor(v, kv_heads_);
    std::vector<float> result(q.size());
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    if (gpu) {
      using nntrainer::cuda::Buffer;
      Buffer qb(q.size() * sizeof(float), q.data()),
        kb(k.size() * sizeof(float), k.data()),
        vb(v.size() * sizeof(float), v.data()),
        ob(result.size() * sizeof(float));
      if (!qb.GetBuffer() || !kb.GetBuffer() || !vb.GetBuffer() ||
          !ob.GetBuffer())
        throw std::runtime_error("FP32 prefix CUDA allocation failed");
      prefixCuda(static_cast<const float *>(qb.GetBuffer()),
                 static_cast<const float *>(kb.GetBuffer()),
                 static_cast<const float *>(vb.GetBuffer()),
                 static_cast<float *>(ob.GetBuffer()), heads_, kv_heads_, dim_,
                 rows, false);
      if (!ob.ReadData(result.data()))
        throw std::runtime_error("prefix CUDA readback failed");
    } else
#endif
    {
      const auto order =
        static_cast<unsigned int>(nntrainer::TStorageOrder::ROW_MAJOR);
      const auto compute_head = [&](size_t h) {
        std::vector<float> scores(static_cast<size_t>(rows) * rows);
        const unsigned int kh = h / (heads_ / kv_heads_);
        // [key, query] storage lets vector softmax reduce over keys while
        // operating on multiple independent queries at once.
        nntrainer::sgemm(order, false, true, rows, rows, dim_,
                         1 / std::sqrt(static_cast<float>(dim_)),
                         k.data() + static_cast<size_t>(kh) * rows * dim_, dim_,
                         q.data() + static_cast<size_t>(h) * rows * dim_, dim_,
                         0.0f, scores.data(), rows);
        nntrainer::softmax_row_inplace<float>(scores.data(), 0, rows, rows);
        nntrainer::sgemm(
          order, true, false, rows, dim_, rows, 1.0f, scores.data(), rows,
          v.data() + static_cast<size_t>(kh) * rows * dim_, dim_, 0.0f,
          result.data() + static_cast<size_t>(h) * rows * dim_, dim_);
      };
#ifdef USE_MKL
      nntrainer::ThreadManager::Global().parallel_for(0, heads_, compute_head);
#else
      for (unsigned int h = 0; h < heads_; ++h)
        compute_head(h);
#endif
    }
    auto &out = ctx.getOutput(0);
    const bool fp32 = out.getDataType() == nntrainer::Tdatatype::FP32;
    auto *bits = reinterpret_cast<uint16_t *>(out.getData());
    for (unsigned int r = 0; r < rows; ++r)
      for (unsigned int h = 0; h < heads_; ++h)
        for (unsigned int d = 0; d < dim_; ++d) {
          const size_t dest = (static_cast<size_t>(r) * heads_ + h) * dim_ + d;
          const float value =
            result[(static_cast<size_t>(h) * rows + r) * dim_ + d];
          if (fp32)
            out.getData<float>()[dest] = value;
          else
            bits[dest] = nntrainer::compute_fp32_to_fp16(value);
        }
  }
  void calcDerivative(nntrainer::RunLayerContext &) override {
    throw std::runtime_error("prefix_attention is inference-only");
  }
  bool supportBackwarding() const override { return false; }
};
} // namespace
std::shared_ptr<ml::train::Layer>
createPrefixAttention(const std::string &name, unsigned int heads,
                      unsigned int kv_heads, unsigned int dim, float theta,
                      const std::shared_ptr<PrefixKV> &cache) {
  return nntrainer::createLayerNode(
    std::make_unique<PrefixAttention>(heads, kv_heads, dim, theta, cache),
    {withKey("name", name), withKey("engine", causallm_engine())});
}
} // namespace causallm
