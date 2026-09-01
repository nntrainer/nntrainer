// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tie_word_embedding.cpp
 * @date   21 May 2025
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#if defined(ENABLE_OPENCL)
// The OpenCL GPU lm_head GEMVs (lmhead_gemv_*_cl). Guarded so the FP32 CPU
// build (enable-opencl=false, which is what the causallm model unittests use)
// compiles tie as a host-only lm_head.
#include <blas_kernels.h>
#endif
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <thread_manager.h>
#include <tie_word_embedding.h>
#include <util_func.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_fc_qs4cx.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>

#if defined(ENABLE_FP16)
namespace {
// NNTR_CUDA_ASYNC guard for the pinned tie-embedding staging buffer: in async
// mode nothing drains the stream per-op, so the NEXT token's host dequant can
// rewrite (or cudaFreeHost) emb_stage while the PREVIOUS token's H2D from the
// same buffer is still in flight -> the consumer kernel reads torn rows
// (measured: word-salad decode under ASYNC=1, coherent under sync). One event
// on the single backend stream marks the most recent staging H2D; stream FIFO
// means "last H2D done" implies every earlier one is done. Skipped during
// graph capture: an in-capture cudaEventSynchronize is illegal and the
// captured H2D is replay-ordered by the graph itself.
cudaEvent_t g_tie_emb_h2d_evt = nullptr;
bool g_tie_emb_h2d_pending = false;

void tie_emb_stage_h2d_record() {
  auto &sm = nntrainer::cuda::StreamManager::Global();
  if (sm.isCapturing())
    return;
  if (g_tie_emb_h2d_evt == nullptr &&
      cudaEventCreateWithFlags(&g_tie_emb_h2d_evt, cudaEventDisableTiming) !=
        cudaSuccess) {
    g_tie_emb_h2d_evt = nullptr;
    cudaGetLastError();
    return;
  }
  if (cudaEventRecord(g_tie_emb_h2d_evt, sm.GetStream()) == cudaSuccess)
    g_tie_emb_h2d_pending = true;
  else
    cudaGetLastError();
}

void tie_emb_stage_h2d_wait() {
  if (!g_tie_emb_h2d_pending ||
      nntrainer::cuda::StreamManager::Global().isCapturing())
    return;
  cudaEventSynchronize(g_tie_emb_h2d_evt);
  g_tie_emb_h2d_pending = false;
}
} // namespace
// The staging buffer, and therefore these two helpers, exist only on the FP16
// path below; defining them in a cuda=on / fp16=off build is an unused-function
// error under -Werror.
#endif
#endif

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum TieWordEmbeddingParams {
  weight,
  bias,
  candidate_weight,
  candidate_hidden_step
};

TieWordEmbedding::TieWordEmbedding() :
  LayerImpl(),
  tieword_embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                          nntrainer::props::Unit(), nntrainer::props::Scale()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void TieWordEmbedding::finalize(nntrainer::InitLayerContext &context) {
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  mode_ = std::get<nntrainer::props::Unit>(tieword_embedding_props).empty()
            ? mode::embedding
            : mode::lm_head;
  if (mode_ == mode::embedding)
    finalize_embedding(context);
  else if (mode_ == mode::lm_head)
    finalize_lmhead(context);
}

void TieWordEmbedding::finalize_embedding(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  // Token-ID input expected (caller responsibility). Input dtype check
  // removed so the layer can sit between an FP32 input layer and FP16
  // activation downstream.

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(tieword_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(tieword_embedding_props);

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void TieWordEmbedding::finalize_lmhead(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(tieword_embedding_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions.
  /// EffDimFlag shouldn't be fixed like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);
  output_dims[0].height(1);

  // @note The logits keep the ACTIVATION dtype on purpose. This was once
  // forced to FP32 because the caller read the logits unconditionally as
  // float*, so an FP16 logits tensor was reinterpreted as FP32 (garbage
  // tokens). That is no longer the case: the whole lm_head chain below is
  // dtype-aware (the Q4_0 / Q6_K / FP32-weight paths all branch on
  // hidden_step's dtype and cast), the device lm_head GEMVs require FP16
  // logits to stay on the device, and incremental_inference() already
  // narrows an FP16 output tensor to the float* the caller expects. Forcing
  // FP32 here silently disables every device path.
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note TieWordEmbedding layer's tensor dim is transposed dim of user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : in_dim.channel(), is_nchw ? unit : 1,
    is_nchw ? in_dim.width() : unit,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "Embedding", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[TieWordEmbeddingParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }
}

void TieWordEmbedding::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, tieword_embedding_props);
  LayerImpl::setProperty(remain_props);
}

void TieWordEmbedding::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {}

void TieWordEmbedding::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  if (mode_ == mode::embedding)
    incremental_forwarding_embedding(context, from, to, training);
  else if (mode_ == mode::lm_head)
    incremental_forwarding_lmhead(context, from, to, training);
  else
    throw std::invalid_argument("lm_head is not supported yet");
}

void TieWordEmbedding::incremental_forwarding_embedding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(tieword_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(tieword_embedding_props);
  float scale =
    std::get<nntrainer::props::Scale>(tieword_embedding_props).empty()
      ? 1.0f
      : std::get<nntrainer::props::Scale>(tieword_embedding_props).get();
  unsigned int _from = from;

  nntrainer::Tensor &weight =
    context.getWeight(weight_idx[TieWordEmbeddingParams::weight]);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  if (!(weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0 ||
        weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K ||
        weight.getDataType() == nntrainer::TensorDim::DataType::FP32))
    throw std::invalid_argument(
      "Tieword embedding is not supported yet for the data type");

  size_t b_size = input_.batch();

  for (size_t b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    int iter = to - from;

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // Device-only activation pool (NNTR_CUDA_DEV_ACT): the output is real
    // device memory (cudaMalloc), NOT host-addressable. The host dequant below
    // cannot store into it directly (segfault). Dequant into a host staging
    // buffer, then push it H2D on the backend stream (ordered before the first
    // GPU layer reads the residual seed). Keeps the CPU off device memory =
    // no page-fault thrash.
    // Persistent + PINNED host staging (was a local std::vector). Under
    // CUDA-graph stream capture a local vector fails twice: (a) a pageable
    // cudaMemcpyAsync is NOT capturable, and (b) the vector is freed when this
    // function returns, but the captured graph REPLAYS afterwards -- it would
    // copy from freed memory => garbage. A process-lifetime pinned
    // (cudaHostAlloc) buffer is capturable and survives the replay. Grows
    // monotonically (decode iter==1; prefill iter<=max_seq_len); single
    // sequence (b_size==1) so one shared buffer is sufficient.
    static _FP16 *emb_stage = nullptr;
    static size_t emb_stage_cap = 0; // capacity in _FP16 elements
    bool emb_dev_only = false;
    if (nntrainer::cuda::engine_selected() &&
        hidden_.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      cudaPointerAttributes pa{};
      emb_dev_only =
        cudaPointerGetAttributes(&pa, batchsliced_hidden.getData<_FP16>()) ==
          cudaSuccess &&
        pa.type == cudaMemoryTypeDevice;
      cudaGetLastError();
      if (emb_dev_only) {
        // Async-mode: the previous token's H2D from this pinned buffer may
        // still be in flight -- wait before the host rewrites or frees it.
        tie_emb_stage_h2d_wait();
        size_t need = (size_t)iter * out_dim;
        if (need > emb_stage_cap) {
          if (emb_stage)
            cudaFreeHost(emb_stage);
          cudaHostAlloc((void **)&emb_stage, need * sizeof(_FP16),
                        cudaHostAllocDefault);
          emb_stage_cap = need;
        }
      }
    }
#endif

    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(iter), [&](size_t i) {
      unsigned int embed_idx = static_cast<unsigned int>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight = weight.getSharedDataTensor(
        out_tensor_dim, static_cast<size_t>(out_dim) * embed_idx);
      nntrainer::Tensor out_tensor = batchsliced_hidden.getSharedDataTensor(
        out_tensor_dim, static_cast<size_t>(out_dim) * (i));

      const auto wt = weight.getDataType();
      if (wt == nntrainer::TensorDim::DataType::Q6_K ||
          wt == nntrainer::TensorDim::DataType::Q4_0) {
        // dequantize_row_q{6_K,4_0} ALWAYS writes out_dim FP32 values. The
        // destination out_tensor is FP16 in an FP16-activation run, so writing
        // FP32 directly would overrun the buffer 2x and corrupt every value
        // (=> garbage row => <pad>). Dequantize into an FP32 scratch, then
        // write into out_tensor with the correct dtype (folding embed scale).
        std::vector<float> tmp(out_dim);
        if (wt == nntrainer::TensorDim::DataType::Q6_K) {
          int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
          nntrainer::dequantize_row_q6_K(
            (void *)((char *)weight.getData<uint8_t>() +
                     (210 * num_blocks_per_row) * embed_idx),
            tmp.data(), out_dim);
        } else {
          int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
          nntrainer::dequantize_row_q4_0(
            (void *)((char *)weight.getData<uint8_t>() +
                     (18 * num_blocks_per_row) * embed_idx),
            tmp.data(), out_dim);
        }
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
          _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
            emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                         out_tensor.getData<_FP16>();
          for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
            o[k] = static_cast<_FP16>(tmp[k] * scale);
#else
          throw std::invalid_argument("FP16 out_tensor requires ENABLE_FP16");
#endif
        } else {
          float *o = out_tensor.getData<float>();
          for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
            o[k] = tmp[k] * scale;
        }
      } else if (wt == nntrainer::TensorDim::DataType::FP32 &&
                 out_tensor.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
        // FP32 embed row -> FP16 activation needs an explicit narrowing cast.
        // copyData byte-copies same-dtype tensors, so an FP32->FP16 copyData
        // writes out_dim*4 bytes into an out_dim*2 buffer => every other value
        // reads as 0 ([0, x, 0, x] corruption => garbage hidden). Mirror the
        // Q6_K/Q4_0 cast path above.
#ifdef ENABLE_FP16
        const float *src = cur_weight.getData<float>();
        _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
          emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                       out_tensor.getData<_FP16>();
        for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
          o[k] = static_cast<_FP16>(src[k] * scale);
#else
        throw std::invalid_argument("FP16 out_tensor requires ENABLE_FP16");
#endif
      } else {
        out_tensor.copyData(cur_weight);
        if (scale != 1.0f) {
          out_tensor.multiply_i(scale);
        }
      }
    });

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // push the host-dequantized embedding rows into the device-only output on
    // the backend stream (ordered before the first GPU layer consumes them).
    if (emb_dev_only) {
      // Windows defaults to a synchronous upload: an asynchronous staging
      // copy here was the measured source of a Windows-only divergence.
      // NNTR_CUDA_EMB_SYNCCOPY=0 restores the async copy; every other
      // platform keeps async.
      static const bool emb_synccopy = []() {
        const char *e = std::getenv("NNTR_CUDA_EMB_SYNCCOPY");
        if (e)
          return e[0] == '1';
#ifdef _WIN32
        return true;
#else
        return false;
#endif
      }();
      if (emb_synccopy &&
          !nntrainer::cuda::StreamManager::Global().isCapturing()) {
        cudaMemcpy(batchsliced_hidden.getData<_FP16>(), emb_stage,
                   (size_t)iter * out_dim * sizeof(_FP16),
                   cudaMemcpyHostToDevice);
      } else {
        cudaMemcpyAsync(batchsliced_hidden.getData<_FP16>(), emb_stage,
                        (size_t)iter * out_dim * sizeof(_FP16),
                        cudaMemcpyHostToDevice,
                        nntrainer::cuda::StreamManager::Global().GetStream());
        tie_emb_stage_h2d_record();
      }
    }
#endif

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

void TieWordEmbedding::incremental_forwarding_lmhead(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  nntrainer::Tensor &weight =
    context.getWeight(weight_idx[TieWordEmbeddingParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    ///@note Since tieword embedding shares the weight with embedding,
    /// the weight is transposed. Thus, the dot product should be consider
    /// this.
    NNTR_THROW_IF(weight.getDataType() == nntrainer::TensorDim::DataType::BCQ,
                  std::invalid_argument)
      << "weight type is not supported for custom tie word embedding layer";

    if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
      ///@note Q4_0 tensor dot does not support trans_in=true for the
      /// embedding-shaped tied weight, so compute each vocab row explicitly.
      const unsigned int hidden_size = input_step.width();
      const unsigned int vocab_size = weight.height();
      NNTR_THROW_IF(weight.width() != hidden_size ||
                      hidden_step.width() != vocab_size,
                    std::invalid_argument)
        << "Q4_0 tie word embedding lmhead has mismatched dimensions";

      const unsigned int num_blocks_per_row = (hidden_size + 32 - 1) / 32;
      const size_t row_size = sizeof(uint16_t) + 16;
      const size_t row_stride = row_size * num_blocks_per_row;
      const uint8_t *weight_data = weight.getData<uint8_t>();
      // dtype-aware I/O: the hidden (input_step) and logits (hidden_step) are
      // FP16 in an FP16-activation run. sdot/dequant operate in fp32, so read
      // the hidden into an fp32 row and write each logit back as the output's
      // dtype (writing fp32 into an fp16 buffer would corrupt it).
      std::vector<float> input_f32;
      const float *input_data;
      if (input_step.getDataType() == nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *in16 = input_step.getData<_FP16>();
        input_f32.resize(hidden_size);
        for (unsigned int k = 0; k < hidden_size; ++k)
          input_f32[k] = static_cast<float>(in16[k]);
        input_data = input_f32.data();
#else
        throw std::invalid_argument("FP16 hidden requires ENABLE_FP16");
#endif
      } else {
        input_data = input_step.getData<float>();
      }
      const bool out_fp16 =
        hidden_step.getDataType() == nntrainer::TensorDim::DataType::FP16;
      float *logits = out_fp16 ? nullptr : hidden_step.getData<float>();
#ifdef ENABLE_FP16
      _FP16 *logits16 = out_fp16 ? hidden_step.getData<_FP16>() : nullptr;
#endif

      auto &tm = nntrainer::ThreadManager::Global();
      const unsigned int compute_thread_num = tm.getComputeThreadCount();
      const unsigned int thread_num =
        compute_thread_num == 0 ? 1 : compute_thread_num;
      tm.parallel_for(0, static_cast<size_t>(thread_num), [=](size_t t) {
        const unsigned int start = (t * vocab_size) / thread_num;
        const unsigned int end = ((t + 1) * vocab_size) / thread_num;
        std::vector<float> dequant_row(hidden_size);

        for (unsigned int row = start; row < end; ++row) {
          nntrainer::dequantize_row_q4_0(
            static_cast<const void *>(weight_data + row_stride * row),
            dequant_row.data(), hidden_size);
          const float val =
            nntrainer::sdot(hidden_size, input_data, 1, dequant_row.data(), 1);
#ifdef ENABLE_FP16
          if (out_fp16)
            logits16[row] = static_cast<_FP16>(val);
          else
#endif
            logits[row] = val;
        }
      });
    } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
      // Q6_K manual lm_head. Mirror the Q4_0 path: dequant each vocab row to
      // fp32, then sdot against the (single) input row. Avoids Tensor::dot,
      // which can crash on gpu-context-allocated tensors when this layer is
      // registered on the OpenCL context.
      const unsigned int hidden_size = input_step.width();
      const unsigned int vocab_size = weight.height();
      NNTR_THROW_IF(weight.width() != hidden_size ||
                      hidden_step.width() != vocab_size,
                    std::invalid_argument)
        << "Q6_K tie word embedding lmhead has mismatched dimensions";

      const unsigned int num_blocks_per_row = (hidden_size + 256 - 1) / 256;
      const size_t row_stride = 210 * num_blocks_per_row;
      const uint8_t *weight_data = weight.getData<uint8_t>();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // engine=cuda GPU Q6_K lm_head: reads the device FP16 hidden + (managed)
      // Q6_K weight directly and writes FP16 logits to the device output -- no
      // host bounce, so it works with a device-only activation pool
      // (NNTR_CUDA_DEV_ACT) where the host GEMV below faults on the device-only
      // hidden/logits. Gated on FP16 in/out that are device-resident; falls
      // through to the host loop otherwise (OpenCL/CPU unaffected).
      if (input_step.getDataType() == nntrainer::TensorDim::DataType::FP16 &&
          hidden_step.getDataType() == nntrainer::TensorDim::DataType::FP16 &&
          (hidden_size % 256) == 0) {
#ifdef ENABLE_FP16
        const _FP16 *hin = input_step.getData<_FP16>();
        _FP16 *hout = hidden_step.getData<_FP16>();
        const bool dev = hin && nntrainer::cuda::dev_accessible(hin);
        if (dev && nntrainer::cuda::lmhead_gemv_q6_k_cuda(
                     weight_data, reinterpret_cast<const unsigned short *>(hin),
                     reinterpret_cast<unsigned short *>(hout), (int)vocab_size,
                     (int)hidden_size))
          return;
#endif
      }
#endif

      // dtype-aware I/O (see Q4_0 path above): FP16 hidden -> fp32 row; logits
      // written back as the output tensor's dtype.
      std::vector<float> input_f32;
      const float *input_data;
      if (input_step.getDataType() == nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *in16 = input_step.getData<_FP16>();
        input_f32.resize(hidden_size);
        for (unsigned int k = 0; k < hidden_size; ++k)
          input_f32[k] = static_cast<float>(in16[k]);
        input_data = input_f32.data();
#else
        throw std::invalid_argument("FP16 hidden requires ENABLE_FP16");
#endif
      } else {
        input_data = input_step.getData<float>();
      }
      const bool out_fp16 =
        hidden_step.getDataType() == nntrainer::TensorDim::DataType::FP16;
      float *logits = out_fp16 ? nullptr : hidden_step.getData<float>();
#ifdef ENABLE_FP16
      _FP16 *logits16 = out_fp16 ? hidden_step.getData<_FP16>() : nullptr;
#endif

      // Decode lm_head on the GPU: the host loop below streams the whole Q6_K
      // table through the CPU every token, which is the dominant decode cost.
      // Gated to GPU runs (NNTR_FC_INT8_GPU, the canonical GPU-FC lever) with
      // NNTR_LMHEAD_GPU=0/1 as an explicit kill switch / opt-in; falls back to
      // the host loop on any failure. The logits differ from the host loop
      // only in fp32 summation order, so greedy token-ID equality is the
      // validation gate.
      static const int lmhead_gpu = []() {
        if (const char *e = std::getenv("NNTR_LMHEAD_GPU"))
          return std::atoi(e);
        // Track the GPU-FC default: the lm_head GEMV is the dominant decode
        // op, so leaving it off when NNTR_FC_INT8_GPU is unset costs about
        // 40% of decode. Default on; =0 disables.
        const char *fc = std::getenv("NNTR_FC_INT8_GPU");
        return (!fc || std::atoi(fc) != 0) ? 1 : 0;
      }();
      bool gpu_done = false;
#if defined(ENABLE_OPENCL)
      // GPU Q6_K lm_head GEMV; falls through to the host loop (gpu_done stays
      // false) on the no-OpenCL build.
      if (lmhead_gpu != 0 && (hidden_size % 256) == 0) {
        std::vector<float> logits_f32(vocab_size);
        gpu_done = nntrainer::lmhead_gemv_q6_k_cl(
          weight_data, input_data, logits_f32.data(), vocab_size, hidden_size);
        if (gpu_done) {
#ifdef ENABLE_FP16
          if (out_fp16) {
            for (unsigned int v = 0; v < vocab_size; ++v)
              logits16[v] = static_cast<_FP16>(logits_f32[v]);
          } else
#endif
          {
            std::memcpy(logits, logits_f32.data(), sizeof(float) * vocab_size);
          }
        }
      }
#endif
      (void)lmhead_gpu;

      if (!gpu_done) {
        auto &tm = nntrainer::ThreadManager::Global();
        const unsigned int compute_thread_num = tm.getComputeThreadCount();
        const unsigned int thread_num =
          compute_thread_num == 0 ? 1 : compute_thread_num;
        tm.parallel_for(0, static_cast<size_t>(thread_num), [=](size_t t) {
          const unsigned int start = (t * vocab_size) / thread_num;
          const unsigned int end = ((t + 1) * vocab_size) / thread_num;
          std::vector<float> dequant_row(hidden_size);

          for (unsigned int row = start; row < end; ++row) {
            nntrainer::dequantize_row_q6_K(
              static_cast<const void *>(weight_data + row_stride * row),
              dequant_row.data(), hidden_size);
            const float val = nntrainer::sdot(hidden_size, input_data, 1,
                                              dequant_row.data(), 1);
#ifdef ENABLE_FP16
            if (out_fp16)
              logits16[row] = static_cast<_FP16>(val);
            else
#endif
              logits[row] = val;
          }
        });
      }
    } else if (weight.getDataType() == nntrainer::TensorDim::DataType::FP32) {
      // Unquantized FP32 (passthrough) lm_head weight. Q6_K loses ~1.66 logit
      // on the first-token argmax (the <think> vs garbage decision => a garbage
      // "noise prefix" on Qwen3 thinking models); use a high-precision FP32-
      // weight GPU GEMV (fp32 W x fp16 act, fp32 accumulate) that matches the
      // HF reference. Falls back to the generic Tensor::dot on any failure.
      const unsigned int hidden_size = input_step.width();
      const unsigned int vocab_size = weight.height();
      const bool out_fp16 =
        hidden_step.getDataType() == nntrainer::TensorDim::DataType::FP16;
      static const int lmhead_gpu_fp32 = []() {
        if (const char *e = std::getenv("NNTR_LMHEAD_GPU"))
          return std::atoi(e);
        const char *fc = std::getenv("NNTR_FC_INT8_GPU");
        return (fc && std::atoi(fc) != 0) ? 1 : 0;
      }();
      bool gpu_done = false;
      // lmhead_gemv_fp32w_cl is only declared/defined under ENABLE_OPENCL
      // (blas_kernels.h is included behind that guard above). Requiring
      // ENABLE_OPENCL here too keeps the FP16-but-no-OpenCL build (e.g. the
      // CPU/HTP libnntrainer) compiling: gpu_done stays false and the host
      // lm_head path below runs.
#if defined(ENABLE_FP16) && defined(ENABLE_OPENCL)
      if (lmhead_gpu_fp32 != 0 &&
          input_step.getDataType() == nntrainer::TensorDim::DataType::FP16 &&
          weight.width() == hidden_size && hidden_step.width() == vocab_size) {
        std::vector<float> logits_f32(vocab_size);
        gpu_done = nntrainer::lmhead_gemv_fp32w_cl(
          static_cast<const void *>(weight.getData<float>()),
          static_cast<const void *>(input_step.getData<_FP16>()),
          logits_f32.data(), vocab_size, hidden_size);
        if (gpu_done) {
          if (out_fp16) {
            _FP16 *o = hidden_step.getData<_FP16>();
            for (unsigned int v = 0; v < vocab_size; ++v)
              o[v] = static_cast<_FP16>(logits_f32[v]);
          } else {
            std::memcpy(hidden_step.getData<float>(), logits_f32.data(),
                        sizeof(float) * vocab_size);
          }
        }
      }
#endif
      if (!gpu_done) {
        // All-FP32 case (FP32 weight AND FP32 act/logits): raw CPU BLAS on
        // the already host-lowered buffers instead of Tensor::dot. dot()
        // dispatches the tensors' ATTACHED op table, which on an engine=gpu
        // graph is ClComputeOps -- it derives the abstract ComputeOps
        // directly, so every raw-pointer BLAS virtual it does not override
        // (sgemv_fp32 / sgemm_fp32 among them) is a loud not-implemented
        // throw, and the FP32 lm_head died on any GPU-engine run. Same math
        // as dotFloat's sgemm(RowMajor, N, T) call; this function is a
        // designed host-compute boundary, so calling the CPU BLAS entry
        // point directly is the honest expression of what already happens
        // here -- not a silent fallback smuggled into the GPU op table.
        if (input_step.getDataType() == nntrainer::TensorDim::DataType::FP32 &&
            !out_fp16 && weight.width() == hidden_size &&
            hidden_step.width() == vocab_size) {
          const unsigned int rows =
            input_step.batch() * input_step.channel() * input_step.height();
          const unsigned int order = static_cast<unsigned int>(
            ml::train::TensorDim::StorageOrder::ROW_MAJOR);
          nntrainer::sgemm(order, false, true, rows, vocab_size, hidden_size,
                           1.0f, input_step.getData<float>(), hidden_size,
                           weight.getData<float>(), hidden_size, 0.0f,
                           hidden_step.getData<float>(), vocab_size);
        } else {
          input_step.dot(weight, hidden_step, false, true);
        }
      }
    } else {
      input_step.dot(weight, hidden_step, false, true);
    }

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[TieWordEmbeddingParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void TieWordEmbedding::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void TieWordEmbedding::calcGradient(nntrainer::RunLayerContext &context) {}

void TieWordEmbedding::exportTo(nntrainer::Exporter &exporter,
                                const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(tieword_embedding_props, method, this);
}

void TieWordEmbedding::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  if (mode_ == mode::embedding) {
    in_dim.width(height);
  } else {
    in_dim.height(height);
  }
  out_dim.height(height);

  context.updateInput(SINGLE_INOUT_IDX, in_dim);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

void TieWordEmbedding::read(
  std::ifstream &file, nntrainer::RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {

  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      /// @note shared weights are only be read at the first acecss
      if (context.isGradientFirstAccess(i)) {
        context.getWeight(i).read(file, start_offset, read_from_offset);
        if (context.isMixedPrecision(i) && trainable &&
            !context.getWeightFP32(i).empty()) {
          context.getWeightFP32(i).copyData(context.getWeight(i));
        }
      }
    }
  }
}

void TieWordEmbedding::read(
  nntrainer::ReadSource src, nntrainer::RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {

  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      /// @note shared weights are only be read at the first acecss
      if (context.isGradientFirstAccess(i)) {
        context.getWeight(i).read(src, start_offset, read_from_offset, file_fd);
        if (context.isMixedPrecision(i) && trainable &&
            !context.getWeightFP32(i).empty()) {
          context.getWeightFP32(i).copyData(context.getWeight(i));
        }
      }
    }
  }
}

void TieWordEmbedding::save(std::ofstream &file,
                            nntrainer::RunLayerContext &run_context,
                            bool opt_var, ml::train::ExecutionMode mode,
                            bool trainable,
                            nntrainer::TensorDim::DataType dtype,
                            ml::train::ISA target_isa) const {
  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    // @note shared weights are only be saved at the first access
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientFirstAccess(i)) {
        auto &weight = run_context.getWeight(i);
        if (dtype == nntrainer::TensorDim::DataType::NONE ||
            weight.getDataType() == dtype)
          weight.save(file);
        else {
          NNTR_THROW_IF(weight.getDataType() !=
                          nntrainer::TensorDim::DataType::FP32,
                        std::runtime_error)
            << "Save with quantization only supports for FP32 weight.";
          ///@note The codelines below can be replaced with quantizer's
          /// quantize()
          nntrainer::TensorDim dim = weight.getDim();
          unsigned int K = dim.height();
          unsigned int N = dim.width();

          if (dtype == nntrainer::TensorDim::DataType::Q4_0) {
            if (K == 1) {
              weight.save(file);
            } else {
              NNTR_THROW_IF(N % 32 != 0, std::invalid_argument)
                << "Q4_0 embedding quantization requires width to be "
                   "divisible by 32, but got width="
                << N;
              nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                             {nntrainer::Tformat::NCHW, dtype});
              nntrainer::quantize_q4_0(weight.getData<float>(),
                                       quant_weight.getData<uint8_t>(), K, N,
                                       nullptr);
              quant_weight.save(file);
            }
          } else if (dtype == nntrainer::TensorDim::DataType::Q6_K) {
            //////////////////////////////////////////////////////////////////
            ///@note Please note that Embedding layer doesn't need to be
            /// transposed!
            //////////////////////////////////////////////////////////////////
            nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                           {nntrainer::Tformat::NCHW, dtype});
            nntrainer::quantize_q6_K(weight.getData<float>(),
                                     quant_weight.getData<uint8_t>(), K, N,
                                     nullptr);
            quant_weight.save(file);
          } else {
            NNTR_THROW_IF(true, std::runtime_error)
              << "This dtype is not supported in save with quantization";
          }
        }
      }
    }
  }
}

} // namespace nntrainer
