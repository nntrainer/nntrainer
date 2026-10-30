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

#include <acti_func.h>
#include <cpu_backend.h>
#include <htp_tie_word_embedding.h>
#include <layer_context.h>
#include <nntr_htp_decode_hook.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <thread_manager.h>
#include <util_func.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum HtpTieWordEmbeddingParams {
  weight,
  bias,
  candidate_weight,
  candidate_hidden_step,
  norm_gamma
};

HtpTieWordEmbedding::HtpTieWordEmbedding() :
  LayerImpl(),
  tieword_embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                          nntrainer::props::Unit(), nntrainer::props::Scale(),
                          props::InNorm(), nntrainer::props::Epsilon(),
                          props::Softcap()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void HtpTieWordEmbedding::finalize(nntrainer::InitLayerContext &context) {
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

void HtpTieWordEmbedding::finalize_embedding(
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

  weight_idx[HtpTieWordEmbeddingParams::weight] = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void HtpTieWordEmbedding::finalize_lmhead(
  nntrainer::InitLayerContext &context) {
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

  // The lm_head output is the logits; generate() reads them as float* for
  // argmax/sampling. Force FP32 regardless of the activation dtype — under FP16
  // activation an FP16 logits tensor is reinterpreted as FP32 and produces
  // garbage tokens. (The Q6_K/Q4_0 lmhead matmul writes FP32 directly; the FP16
  // activation is cast up to FP32 before the dot in incremental_forwarding.)
  output_dims[0].setTensorType(
    {context.getFormat(), nntrainer::TensorDim::DataType::FP32});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note HtpTieWordEmbedding layer's tensor dim is transposed dim of
  /// user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : in_dim.channel(), is_nchw ? unit : 1,
    is_nchw ? in_dim.width() : unit,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[HtpTieWordEmbeddingParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "Embedding", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[HtpTieWordEmbeddingParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  in_norm = std::get<props::InNorm>(tieword_embedding_props).get();
  softcap = std::get<props::Softcap>(tieword_embedding_props).get();
  NNTR_THROW_IF(softcap < 0.0f, std::invalid_argument)
    << "lm head: softcap must be >= 0";
  if (in_norm) {
    // FP32 whatever the weights are, as the rms_norm layer requests it
    weight_idx[HtpTieWordEmbeddingParams::norm_gamma] = context.requestWeight(
      ml::train::TensorDim(
        1, 1, 1, in_dim.width(),
        ml::train::TensorDim::TensorType(context.getFormat(),
                                         nntrainer::TensorDim::DataType::FP32)),
      nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  }
}

void HtpTieWordEmbedding::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, tieword_embedding_props);
  LayerImpl::setProperty(remain_props);
}

void HtpTieWordEmbedding::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {}

void HtpTieWordEmbedding::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  if (mode_ == mode::embedding)
    incremental_forwarding_embedding(context, from, to, training);
  else if (mode_ == mode::lm_head)
    incremental_forwarding_lmhead(context, from, to, training);
  else
    throw std::invalid_argument("lm_head is not supported yet");
}

void HtpTieWordEmbedding::incremental_forwarding_embedding(
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
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::weight]);
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

    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(iter), [&](size_t i) {
      unsigned int embed_idx = static_cast<unsigned int>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight = weight.getSharedDataTensor(
        out_tensor_dim, static_cast<size_t>(out_dim) * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K ||
          weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        const bool is_q6k =
          weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K;
        const int blk = is_q6k ? 256 : 32;
        const int bytes_per_blk = is_q6k ? 210 : 18;
        const int num_blocks_per_row = (weight.width() + blk - 1) / blk;
        const void *src =
          (void *)((char *)weight.getData<uint8_t>() +
                   (bytes_per_blk * num_blocks_per_row) * embed_idx);

        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          if (is_q6k)
            nntrainer::dequantize_row_q6_K(src, out_tensor.getData(), out_dim);
          else
            nntrainer::dequantize_row_q4_0(src, out_tensor.getData(), out_dim);
        } else {
          // dequantize_row_* writes FP32. Under a non-FP32 (e.g. FP16)
          // activation, writing straight into out_tensor would scribble each
          // 4-byte float across the narrower slots and corrupt the embedding
          // (multilingual-garbage symptom). Dequantize into an FP32 temp then
          // cast into the activation dtype.
          nntrainer::TensorDim fp32_dim(
            {1, 1, 1, out_dim}, nntrainer::TensorDim::TensorType(
                                  out_tensor_dim.getFormat(),
                                  nntrainer::TensorDim::DataType::FP32));
          nntrainer::Tensor tmp(fp32_dim, true);
          if (is_q6k)
            nntrainer::dequantize_row_q6_K(src, tmp.getData(), out_dim);
          else
            nntrainer::dequantize_row_q4_0(src, tmp.getData(), out_dim);
          out_tensor.copyData(tmp);
        }
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    });

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

/**
 * @brief Builds the blocked twin of the tied weight, or leaves it empty.
 *
 * repack_q4_0 reads exactly what the tied weight already is -- nrow rows of
 * k canonical Q4_0 values -- so nrow is the vocabulary and k the hidden
 * size, and there is no transpose. It refuses a shape it cannot interleave
 * (nrow % 4, k % 8), and block counts differ per ISA, so the size is taken
 * from the weight rather than computed: a block_q4_0x4 holds four
 * block_q4_0 and the total is unchanged.
 *
 * Every failure here leaves lmhead_blocked_ empty and the caller on the
 * row-wise path, which is slow but right. This runs once per process.
 */
void HtpTieWordEmbedding::prepareLmhead(nntrainer::RunLayerContext &context) {
  if (mode_ != mode::lm_head || lmhead_blocked_tried_)
    return;
  const nntrainer::Tensor &weight =
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::weight]);
  if (weight.getDataType() != nntrainer::TensorDim::DataType::Q4_0)
    return;
  lmhead_blocked_tried_ = true;
  buildLmheadBlocked(weight, weight.height(), weight.width());
}

bool HtpTieWordEmbedding::placeLmheadOnAccelerator(
  nntrainer::RunLayerContext &context) {
  if (mode_ != mode::lm_head || lmhead_blocked_tried_)
    return false;
  nntrainer::ComputeOps *ops = context.getComputeOps();
  const nntrainer::Tensor &weight =
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::weight]);
  if (ops == nullptr ||
      weight.getDataType() != nntrainer::TensorDim::DataType::Q4_0)
    return false;
  if (!ops->lm_head_q4_0_prepare(weight.getData<uint8_t>(), weight.width(),
                                 weight.height()))
    return false;
  // The calls run there; the blocked twin (a second copy of the weight,
  // 396 MiB here) would only serve a CPU fallback, which the per-row path
  // still covers. Measured before this: the placement ran inside the
  // first prefill, 739 ms, and the twin stayed resident beside it (doc 57
  // section 9.10).
  lmhead_blocked_tried_ = true;
  return true;
}

void HtpTieWordEmbedding::buildLmheadBlocked(const nntrainer::Tensor &weight,
                                             unsigned int vocab_size,
                                             unsigned int hidden_size) {
  // The interleave's own preconditions (nntr_repack_q4_0_to_q4_0_4_bl
  // returns -1 rather than throwing, and the ggml wrapper drops that).
  if (vocab_size % 4 != 0 || hidden_size % 8 != 0 || hidden_size % 32 != 0) {
    ml_logw("lm_head %ux%u cannot be interleaved; staying on the per-row "
            "Q4_0 path",
            vocab_size, hidden_size);
    return;
  }

  const size_t bytes = weight.bytes();
  void *src =
    const_cast<void *>(static_cast<const void *>(weight.getData<uint8_t>()));
  if (src == nullptr || bytes == 0)
    return;

  try {
    lmhead_blocked_.resize(bytes);
  } catch (const std::bad_alloc &) {
    // 75 MB for this model. Worth reporting, not worth failing a load over.
    ml_logw("lm_head: no room for a %zu byte blocked weight; staying on the "
            "per-row Q4_0 path",
            bytes);
    lmhead_blocked_.clear();
    return;
  }

  nntrainer::repack_q4_0(lmhead_blocked_.data(), src, bytes, vocab_size,
                         hidden_size);
  ml_logd("lm_head: blocked %ux%u Q4_0 weight, %zu bytes", vocab_size,
          hidden_size, bytes);
}

std::vector<unsigned int> HtpTieWordEmbedding::ppl_targets_;
double HtpTieWordEmbedding::ppl_nll_ = 0.0;
unsigned int HtpTieWordEmbedding::ppl_count_ = 0;

void HtpTieWordEmbedding::setPplTargets(const std::vector<unsigned int> &ids) {
  ppl_targets_ = ids;
  ppl_nll_ = 0.0;
  ppl_count_ = 0;
}

bool HtpTieWordEmbedding::takePpl(double &nll_sum, unsigned int &count) {
  if (ppl_targets_.empty())
    return false;
  nll_sum = ppl_nll_;
  count = ppl_count_;
  ppl_targets_.clear();
  ppl_nll_ = 0.0;
  ppl_count_ = 0;
  return true;
}

double HtpTieWordEmbedding::nllOf(const float *logits, unsigned int vocab,
                                  unsigned int target) {
  float mx = logits[0];
  for (unsigned int v = 1; v < vocab; ++v)
    mx = std::max(mx, logits[v]);
  double sum = 0.0;
  for (unsigned int v = 0; v < vocab; ++v)
    sum += std::exp(static_cast<double>(logits[v]) - mx);
  return std::log(sum) + mx - static_cast<double>(logits[target]);
}

void HtpTieWordEmbedding::incremental_forwarding_lmhead(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  nntrainer::Tensor &weight =
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::weight]);

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

  // One input row's logits. A lambda so the NNTR_PPL pass below can run
  // it over every prefill row, not only the last.
  auto logits_of = [&](nntrainer::Tensor &input_step,
                       nntrainer::Tensor &hidden_step) {
    ///@note Since tieword embedding shares the weight with embedding,
    /// the weight is transposed. Thus, the dot product should be consider
    /// this.
    NNTR_THROW_IF(weight.getDataType() == nntrainer::TensorDim::DataType::BCQ,
                  std::invalid_argument)
      << "weight type is not supported for custom tie word embedding layer";

    if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
      ///@note The tied (embedding-shaped) weight is [vocab, hidden]. The fused
      /// Q6_K GEMV computes logits[v] = input . weight[v] row-wise, which needs
      /// no data transpose, and is ~2x faster per decode token than a per-row
      /// dequantize+sdot loop. The lmhead output is forced FP32 (finalize);
      /// cast a FP16 activation up to FP32 first so FloatTensor::dotQnK writes
      /// FP32 logits directly (generate() reads them as float*).
      nntrainer::Tensor input_fp32 =
        (input_step.getDataType() == nntrainer::TensorDim::DataType::FP32)
          ? input_step
          : input_step.clone(nntrainer::TensorDim::DataType::FP32);
      input_fp32.dot(weight, hidden_step, false, true);
    } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
      ///@note The tied weight is stored as canonical Q4_0 rows so embedding
      /// lookup can address a token directly. Quantize the activation once and
      /// compute all vocabulary logits with the row-wise Q4_0/Q8_0 SIMD GEMV.
      const unsigned int hidden_size = input_step.width();
      const unsigned int vocab_size = weight.height();
      NNTR_THROW_IF(weight.width() != hidden_size ||
                      hidden_step.width() != vocab_size,
                    std::invalid_argument)
        << "Q4_0 tie word embedding lmhead has mismatched dimensions";

      const uint8_t *weight_data = weight.getData<uint8_t>();

      // The row-wise GEMV accepts FP32 activations and writes FP32 logits.
      nntrainer::Tensor input_fp32 =
        (input_step.getDataType() == nntrainer::TensorDim::DataType::FP32)
          ? input_step
          : input_step.clone(nntrainer::TensorDim::DataType::FP32);
      const float *input_data = input_fp32.getData<float>();
      float *logits = hidden_step.getData<float>();

      // The blocked twin, built once. See lmhead_blocked_'s comment for why
      // a tied weight arrives unrepacked and what that costs.
      if (!lmhead_blocked_tried_) {
        lmhead_blocked_tried_ = true;
        buildLmheadBlocked(weight, vocab_size, hidden_size);
      }

      if (!lmhead_blocked_.empty()) {
        // Same call the fully_connected layers make (float_tensor.cpp's
        // Q4_0 branch): A is 1 x K, B is the blocked N x K weight, C is
        // 1 x N. The tied weight is already [vocab, hidden], which is the
        // [N, K] this wants, so nothing is transposed anywhere.
        nntrainer::gemm_q4_0<float>(1, vocab_size, hidden_size, input_data,
                                    hidden_size, lmhead_blocked_.data(),
                                    vocab_size, logits, vocab_size);
      } else {
        std::vector<char> quantized_activation(
          nntrainer::q4_0_gemv_activation_size(hidden_size));
        nntrainer::quantize_q4_0_gemv_activation(hidden_size, input_data,
                                                 quantized_activation.data());

        auto &tm = nntrainer::ThreadManager::Global();
        constexpr size_t chunks_per_thread = 4;
        const size_t num_chunks = std::max<size_t>(
          1, std::min(static_cast<size_t>(vocab_size),
                      static_cast<size_t>(tm.getComputeThreadCount()) *
                        chunks_per_thread));

        tm.parallel_for(0, num_chunks, [&](size_t chunk) {
          const size_t row_begin = chunk * vocab_size / num_chunks;
          const size_t row_end = (chunk + 1) * vocab_size / num_chunks;
          nntrainer::gemv_q4_0_rowwise_range(row_begin, row_end, hidden_size,
                                             quantized_activation.data(),
                                             weight_data, logits);
        });
      }
    } else {
      input_step.dot(weight, hidden_step, false, true);
    }

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[HtpTieWordEmbeddingParams::bias]);
      hidden_step.add_i(bias);
    }
  };

  // One raw input row's logits with the folded output norm and softcap
  // (doc 57 section 9.7): the accelerator runs the three as one call when
  // it can; otherwise the rms_norm layer's kernel and gamma multiply, the
  // head above, and the logit_softcapping layer's arithmetic, in that
  // layer's order. cap false: the NNTR_PPL rows, scored uncapped as they
  // were before the fold.
  const auto f32 = nntrainer::TensorDim::DataType::FP32;
  const float eps =
    std::get<nntrainer::props::Epsilon>(tieword_embedding_props).get();
  const bool no_bias =
    !std::get<nntrainer::props::DisableBias>(*layer_impl_props).empty() &&
    std::get<nntrainer::props::DisableBias>(*layer_impl_props).get();
  nntrainer::ComputeOps *ops = input_.getOps();
  auto head_of = [&](nntrainer::Tensor &row, nntrainer::Tensor &out, bool cap) {
    const float c = cap ? softcap : 0.0f;
    if (no_bias && ops != nullptr && row.getDataType() == f32 &&
        weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0 &&
        ops->lm_head_q4_0_fp32(
          weight.getData<uint8_t>(), row.width(), out.width(),
          row.getData<float>(),
          in_norm
            ? context
                .getWeight(weight_idx[HtpTieWordEmbeddingParams::norm_gamma])
                .getData<float>()
            : nullptr,
          eps, c, out.getData<float>()))
      return;
    if (in_norm) {
      NNTR_THROW_IF(row.getDataType() != f32, std::invalid_argument)
        << "lm head: in_norm is FP32 only";
      nntrainer::Tensor normed(row.getDim());
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        row.getData<float>(), normed.getData<float>(), 1, row.width(), eps);
      normed.multiply_i(
        context.getWeight(weight_idx[HtpTieWordEmbeddingParams::norm_gamma]));
      logits_of(normed, out);
    } else {
      logits_of(row, out);
    }
    if (c > 0.0f) {
      float *y = out.getData<float>();
      const float inv = 1.0f / c;
      for (size_t i = 0; i < out.size(); ++i)
        y[i] = nntrainer::ActiFunc::tanhFloat<float>(y[i] * inv) * c;
    }
  };

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);
    // [#132 Part B] a decode row's lm_head on the HTP (no bias there, and
    // nothing folded: the resident graph's op is the head alone). [#260]
    // Folded too when the row is resident: the list's final RMSNORM and
    // its LM_HEAD's softcap are the DSP's.
    if (b_size == 1 && to - from == 1 &&
        ((!in_norm && softcap == 0.0f) ||
         nntrainer::htpDecodeRowResident(from)) &&
        input_step.getDataType() == f32 && no_bias &&
        nntrainer::htpDecodeLmHead(
          from, input_step.getData<float>(), input_step.width(),
          hidden_step.getData<float>(), hidden_step.width()))
      continue;
    head_of(input_step, hidden_step, true);
  }

  // NNTR_PPL: every other prefill row through the same lambda into a
  // scratch row, scored against the next prompt token. Row by row, so the
  // bias and every weight type are handled exactly as the model's own
  // row is; ~7 ms a row at this vocabulary, a few seconds a prompt.
  if (is_prefill && !ppl_targets_.empty() && b_size == 1 && to > from + 1) {
    const unsigned int rows = to - from - 1; /* rows that have a target */
    const unsigned int vocab = hidden_dim.width();
    nntrainer::Tensor scratch(hidden_step_dim);
    for (unsigned int t = 0; t < rows; ++t) {
      const size_t pos = static_cast<size_t>(from) + t + 1;
      if (pos >= ppl_targets_.size())
        break;
      nntrainer::Tensor row = input_.getSharedDataTensor(
        input_step_dim, static_cast<size_t>(t) * input_.width(), true);
      head_of(row, scratch, false);
      ppl_nll_ += nllOf(scratch.getData<float>(), vocab, ppl_targets_[pos]);
      ++ppl_count_;
    }
  }
}

void HtpTieWordEmbedding::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void HtpTieWordEmbedding::calcGradient(nntrainer::RunLayerContext &context) {}

void HtpTieWordEmbedding::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(tieword_embedding_props, method, this);
}

void HtpTieWordEmbedding::updateTensorsByInputDimensions(
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

void HtpTieWordEmbedding::read(
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
  } else if (in_norm) {
    // the lm_head's own weight: the folded output norm's gamma, where the
    // output_norm layer's was (the tied weight stays the embedding's)
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::norm_gamma])
      .read(file, start_offset, read_from_offset);
  }
}

void HtpTieWordEmbedding::read(
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
  } else if (in_norm) {
    context.getWeight(weight_idx[HtpTieWordEmbeddingParams::norm_gamma])
      .read(src, start_offset, read_from_offset, file_fd);
  }
}

void HtpTieWordEmbedding::save(std::ofstream &file,
                               nntrainer::RunLayerContext &run_context,
                               bool opt_var, ml::train::ExecutionMode mode,
                               bool trainable,
                               nntrainer::TensorDim::DataType dtype,
                               ml::train::ISA target_isa) const {
  // the lm_head's folded norm gamma, FP32 whatever the target dtype (the
  // rms_norm layer's save)
  if (mode_ == mode::lm_head && in_norm) {
    run_context.getWeight(weight_idx[HtpTieWordEmbeddingParams::norm_gamma])
      .save(file);
    return;
  }
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

#ifdef PLUGGABLE

nntrainer::Layer *create_tie_word_embedding() {
  auto layer = new HtpTieWordEmbedding();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_tie_word_embedding(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_tie_word_embedding,
                                                   destroy_tie_word_embedding};
}

#endif

} // namespace nntrainer
