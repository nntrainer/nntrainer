// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   conv_block_layer.cpp
 * @date   22 September 2026
 * @brief  The LFM2 conv block (in_proj, gate, causal conv1d, gate, out_proj)
 *         as one layer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <conv_block_layer.h>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <q4_0_utils.h>
#include <thread_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum ConvBlockParams { IN_PROJ, CONV_W, OUT_PROJ };
enum ConvBlockTensors { PROJ, GATED, CONV_OUT, STATE };

/** NNTR_CONV_BLOCK_DIFF: run both paths at prefill and print the SNR of
 *  the accelerator's output and conv state against the CPU's, per layer.
 *  NNTR_CONV_BLOCK_SHADOW: same, and hand the model the CPU's values --
 *  the text then says whether the accelerator's numbers are what changed
 *  it. The same two discriminators the MoE path has (NNTR_L2_DIFF /
 *  NNTR_L2_SHADOW), for the same reason: wrong text cannot tell a bug
 *  from a quantization point. */
static bool convBlockDiffEnabled() {
  static const bool on = std::getenv("NNTR_CONV_BLOCK_DIFF") != nullptr;
  return on;
}
static bool convBlockShadowEnabled() {
  static const bool on = std::getenv("NNTR_CONV_BLOCK_SHADOW") != nullptr;
  return on;
}
/** The DSP's per-row u8 rule (hvx_quant_rows_u8_params + the pack), in
 *  f32: min and max over the row with 0 folded in, 255 steps, round to
 *  nearest, then back to f32. Applied to the CPU path's activations it
 *  isolates what the two activation points cost, apart from the weight
 *  recipe (doc 51 section 2.10). */
static void fakeQuantRowsU8(float *x, unsigned int rows, unsigned int k) {
  for (unsigned int r = 0; r < rows; ++r) {
    float *row = x + static_cast<size_t>(r) * k;
    float lo = 0.f, hi = 0.f;
    for (unsigned int j = 0; j < k; ++j) {
      lo = std::min(lo, row[j]);
      hi = std::max(hi, row[j]);
    }
    float s = (hi - lo) / 255.f;
    if (s <= 0.f)
      s = 1e-8f;
    long zp = std::lrint(-lo / s);
    zp = std::max(0L, std::min(255L, zp));
    for (unsigned int j = 0; j < k; ++j) {
      long q = std::lrint(row[j] / s) + zp;
      q = std::max(0L, std::min(255L, q));
      row[j] = static_cast<float>(q - zp) * s;
    }
  }
}

/** Mean over rows of max|x| / rms(x): how far the row's largest value sits
 *  above its typical one. Per-row u8 spends its 255 steps on the range, so
 *  this ratio is what decides how many steps the typical value gets. */
static double outlierRatio(const float *x, unsigned int rows, unsigned int k) {
  double acc = 0.0;
  for (unsigned int r = 0; r < rows; ++r) {
    const float *row = x + static_cast<size_t>(r) * k;
    double ss = 0.0, mx = 0.0;
    for (unsigned int j = 0; j < k; ++j) {
      ss += (double)row[j] * row[j];
      mx = std::max(mx, std::fabs((double)row[j]));
    }
    acc += mx / std::sqrt(ss / k + 1e-30);
  }
  return acc / rows;
}

/* ---- weight-recipe emulation, f32 on the CPU (doc 51 section 2.11) ----
   What each way of quantizing the two projections' weights would cost,
   measured before any of them is built on the DSP. The weights come out
   of the file as Q4_0 (a scale per 32 along K); the DSP path requantizes
   them at load to one int4 scale per output channel. Each recipe below
   takes the dequantized f32 weight, quantizes it its own way, and runs
   the block in f32 with the activations through the DSP's per-row u8
   rule, so the SNRs are comparable to the accelerator's own. */

/** out[t][n] = sum_k x[t][k] * w[n][k]; w is N rows of K, as
 *  Q4_0Utils::dequantizeQ4_0x4 lays it out. */
static void emuGemm(const float *x, unsigned int rows, unsigned int K,
                    const float *w, unsigned int N, float *out) {
  nntrainer::ThreadManager::Global().parallel_for(
    0, static_cast<size_t>(rows), [&](size_t t) {
      const float *xr = x + t * K;
      float *o = out + t * N;
      for (unsigned int n = 0; n < N; ++n) {
        const float *wr = w + static_cast<size_t>(n) * K;
        float acc = 0.f;
        for (unsigned int k = 0; k < K; ++k)
          acc += xr[k] * wr[k];
        o[n] = acc;
      }
    });
}

/** The load-time rule (htp_qs4cx_from_q4_0x4): per output channel,
 *  scale = 15 / (rmax - rmin) with 0 folded in, round, clamp to [-8, 7]. */
static void quantChannelsCur(float *w, unsigned int N, unsigned int K) {
  for (unsigned int n = 0; n < N; ++n) {
    float *row = w + static_cast<size_t>(n) * K;
    float lo = 0.f, hi = 0.f;
    for (unsigned int k = 0; k < K; ++k) {
      lo = std::min(lo, row[k]);
      hi = std::max(hi, row[k]);
    }
    const float scale = (lo == hi) ? 1.f : 15.f / (hi - lo);
    for (unsigned int k = 0; k < K; ++k) {
      long q = std::lrint(row[k] * scale);
      q = std::max(-8L, std::min(7L, q));
      row[k] = static_cast<float>(q) / scale;
    }
  }
}

/** Symmetric per channel: step = max|w| / qmax, clamp to [-qmax, qmax]. */
static void quantChannelsSym(float *w, unsigned int N, unsigned int K,
                             long qmax) {
  for (unsigned int n = 0; n < N; ++n) {
    float *row = w + static_cast<size_t>(n) * K;
    float mx = 0.f;
    for (unsigned int k = 0; k < K; ++k)
      mx = std::max(mx, std::fabs(row[k]));
    const float step = mx > 0.f ? mx / static_cast<float>(qmax) : 1.f;
    for (unsigned int k = 0; k < K; ++k) {
      long q = std::lrint(row[k] / step);
      q = std::max(-qmax, std::min(qmax, q));
      row[k] = static_cast<float>(q) * step;
    }
  }
}

/** Symmetric int4 per (channel, K-group of @a B): what the DSP would see
 *  if each K-group were its own registered weight with its own scale,
 *  dequantized and summed in f32. Symmetric, not the load-time rule: that
 *  rule folds 0 into an asymmetric range without a zero point, and on a
 *  small group whose min and max differ it clips one side -- the first
 *  emulation of these groups used it and read WORSE for smaller groups.
 *  B = 32 is Q4_0's own grid, so its row must land on the act-u8-only
 *  number; that is the check on this emulation. */
static void quantChannelsBlocked(float *w, unsigned int N, unsigned int K,
                                 unsigned int B) {
  for (unsigned int n = 0; n < N; ++n)
    for (unsigned int k0 = 0; k0 < K; k0 += B)
      quantChannelsSym(w + static_cast<size_t>(n) * K + k0, 1,
                       std::min(B, K - k0), 7L);
}

static double snrDb(const float *ref, const float *got, size_t n) {
  double sig = 0.0, err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)ref[i] - (double)got[i];
    sig += (double)ref[i] * (double)ref[i];
    err += d * d;
  }
  return err == 0.0 ? 999.0 : 10.0 * std::log10(sig / err);
}

ConvBlockLayer::ConvBlockLayer() :
  LayerImpl(), conv_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

void ConvBlockLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "conv_block takes exactly one input";
  NNTR_THROW_IF(context.getFormat() != nntrainer::Tformat::NCHW,
                std::invalid_argument)
    << "conv_block supports NCHW only";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  const unsigned int C = std::get<nntrainer::props::Unit>(conv_props).get();

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  const auto &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "conv_block input channel must be 1 (B x 1 x T x W)";
  const unsigned int K = in_dim.width();
  const nntrainer::TensorDim::TensorType act_type(
    context.getFormat(), context.getActivationDataType());
  const nntrainer::TensorDim::TensorType w_type(context.getFormat(),
                                                context.getWeightDataType());
  const nntrainer::TensorDim::TensorType f32_type(
    context.getFormat(), ml::train::TensorDim::DataType::FP32);

  // out_proj's width is the residual width: the block adds into it.
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.setTensorType(act_type);
  context.setOutputDimensions({out_dim});

  // The file's order: in_proj, conv, out_proj. Shapes as the three layers
  // this replaces request them ([in, unit] for the FCs, [1, 1, 3, C] FP32
  // for the conv), so the same bytes load either way.
  nntrainer::TensorDim w_in(1, 1, K, 3 * C, w_type, 0b0011);
  weight_idx[IN_PROJ] = context.requestWeight(
    w_in, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "in_proj", true);
  nntrainer::TensorDim w_conv({1, 1, KERNEL_SIZE, C}, f32_type);
  weight_idx[CONV_W] = context.requestWeight(
    w_conv, nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    0.0f, 0.0f, "conv", false);
  nntrainer::TensorDim w_out(1, 1, C, K, w_type, 0b0011);
  weight_idx[OUT_PROJ] = context.requestWeight(
    w_out, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "out_proj", true);

  // The CPU path's intermediates, sized for the graph's input length and
  // sliced per step below; the fused call never touches them.
  nntrainer::TensorDim proj(in_dim.batch(), 1, in_dim.height(), 3 * C,
                            act_type);
  tensor_idx[PROJ] =
    context.requestTensor(proj, "proj", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  nntrainer::TensorDim mid(in_dim.batch(), 1, in_dim.height(), C, act_type);
  tensor_idx[GATED] =
    context.requestTensor(mid, "gated", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  tensor_idx[CONV_OUT] =
    context.requestTensor(mid, "conv_out", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  // The conv state, as causal_conv1d_layer keeps it: [x_{t-2}, x_{t-1}].
  nntrainer::TensorDim state({in_dim.batch(), 1, KERNEL_SIZE - 1, C}, f32_type);
  tensor_idx[STATE] =
    context.requestTensor(state, "conv_state", nntrainer::Initializer::ZEROS,
                          false, nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void ConvBlockLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void ConvBlockLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

void ConvBlockLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  incremental_forwarding(context, 0,
                         context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void ConvBlockLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "conv_block: invalid range from=" << from << " to=" << to;
  nntrainer::Tensor &in_w = context.getWeight(weight_idx[IN_PROJ]);
  nntrainer::Tensor &conv_w = context.getWeight(weight_idx[CONV_W]);
  nntrainer::Tensor &out_w = context.getWeight(weight_idx[OUT_PROJ]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &state_t = context.getTensor(tensor_idx[STATE]);

  // Batch 1, rows at offset 0 of the step -- where the model places the
  // current token(s), as causal_conv1d_layer and dense_ffn_layer read them.
  const unsigned int rows = to - from;
  nntrainer::TensorDim in_step_dim = input_.getDim();
  in_step_dim.batch(1);
  in_step_dim.height(rows);
  nntrainer::Tensor in_step = input_.getSharedDataTensor(in_step_dim, 0, true);
  nntrainer::TensorDim out_step_dim = output_.getDim();
  out_step_dim.batch(1);
  out_step_dim.height(rows);
  nntrainer::Tensor out_step =
    output_.getSharedDataTensor(out_step_dim, 0, true);

  const unsigned int K = in_step_dim.width();
  const unsigned int C = conv_w.width();
  const unsigned int N = out_w.width();
  const float *w_ptr = conv_w.getData<float>();
  float *state = state_t.getData<float>();

  // The whole block as one accelerator call at prefill; it hands back the
  // conv state decode continues from. Decode's single row stays on the
  // CPU for the reason every FC gate has (accelerates_q4_0_at_m1): one
  // row cannot amortize the call.
  const auto q4 = ml::train::TensorDim::DataType::Q4_0;
  auto *ops = in_step.getOps();
  const bool use_htp = rows > 1 && ops != nullptr &&
                       ops->supports_gemm_q4_0_conv_block_fp32() &&
                       in_w.getDataType() == q4 && out_w.getDataType() == q4;
  const bool compare =
    use_htp && (convBlockDiffEnabled() || convBlockShadowEnabled());
  if (use_htp && !compare) {
    ops->gemm_q4_0_conv_block_fp32(
      in_w.getData<char>(), w_ptr, out_w.getData<char>(),
      in_step.getData<float>(), out_step.getData<float>(), state, rows, K, C,
      N);
    return;
  }

  // What createConvBlock's layers computed, with their kernels: in_proj
  // dot, a * c, the causal conv (decode's kernel keeps the state, the
  // prefill one is followed by the same state save), b * conv, out_proj
  // dot. Each multiply is one f32 operation, so the split and the two
  // custom_multiply layers this folds are byte for byte.
  nntrainer::TensorDim proj_dim = context.getTensor(tensor_idx[PROJ]).getDim();
  proj_dim.batch(1);
  proj_dim.height(rows);
  nntrainer::Tensor proj =
    context.getTensor(tensor_idx[PROJ]).getSharedDataTensor(proj_dim, 0, true);
  nntrainer::TensorDim mid_dim = context.getTensor(tensor_idx[GATED]).getDim();
  mid_dim.batch(1);
  mid_dim.height(rows);
  nntrainer::Tensor gated =
    context.getTensor(tensor_idx[GATED]).getSharedDataTensor(mid_dim, 0, true);
  nntrainer::Tensor conv_out = context.getTensor(tensor_idx[CONV_OUT])
                                 .getSharedDataTensor(mid_dim, 0, true);

  // Under compare the projections go through the CPU Q4_0 GEMM directly:
  // with engine=htp on this layer, dot() would route them to the HTP FC
  // path and the "reference" would carry the same quantization points as
  // the fused call (the first DIFF run measured that as 150 dB and proved
  // the kernel, not the numerics -- doc 51 section 2.9).
  if (compare) {
    nntrainer::gemm_q4_0<float>(rows, 3 * C, K, in_step.getData<float>(), K,
                                in_w.getData<char>(), 3 * C,
                                proj.getData<float>(), 3 * C);
  } else {
    in_step.dot(in_w, proj, false, false);
  }

  const float *p = proj.getData<float>();
  float *g = gated.getData<float>();
  float *y = conv_out.getData<float>();
  auto gate_pre = [&](size_t r) {
    const float *a = p + r * 3 * C;
    const float *c = a + 2 * C;
    float *gr = g + r * C;
    for (unsigned int j = 0; j < C; ++j)
      gr[j] = a[j] * c[j];
  };
  auto gate_post = [&](size_t r) {
    const float *b = p + r * 3 * C + C;
    float *yr = y + r * C;
    for (unsigned int j = 0; j < C; ++j)
      yr[j] = b[j] * yr[j];
  };
  if (rows == 1) {
    gate_pre(0);
    nntrainer::causal_depthwise_conv1d_k3_decode(g, w_ptr, state, y, C);
    gate_post(0);
  } else {
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(rows), gate_pre);
    nntrainer::causal_depthwise_conv1d_k3(g, w_ptr, nullptr, y, 1, rows, C);
    std::memcpy(state, g + static_cast<size_t>(rows - 2) * C,
                C * sizeof(float));
    std::memcpy(state + C, g + static_cast<size_t>(rows - 1) * C,
                C * sizeof(float));
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(rows), gate_post);
  }

  if (compare) {
    nntrainer::gemm_q4_0<float>(rows, N, C, y, C, out_w.getData<char>(), N,
                                out_step.getData<float>(), N);
  } else {
    conv_out.dot(out_w, out_step, false, false);
  }

  if (compare) {
    // The CPU path above is the reference; the accelerator recomputes
    // the same step from the same input into its own buffers.
    std::vector<float> h_out(static_cast<size_t>(rows) * N);
    std::vector<float> h_state(static_cast<size_t>(2) * C);
    ops->gemm_q4_0_conv_block_fp32(
      in_w.getData<char>(), w_ptr, out_w.getData<char>(),
      in_step.getData<float>(), h_out.data(), h_state.data(), rows, K, C, N);
    // The activation points alone: the CPU path again with x and y put
    // through the DSP's per-row u8 rule (weights still Q4_0). Its SNR
    // against the reference, next to the accelerator's, says whether the
    // loss is the activations or the weight recipe.
    std::vector<float> x_fq(in_step.getData<float>(),
                            in_step.getData<float>() +
                              static_cast<size_t>(rows) * K);
    const double x_ratio = outlierRatio(x_fq.data(), rows, K);
    fakeQuantRowsU8(x_fq.data(), rows, K);
    std::vector<float> p_fq(static_cast<size_t>(rows) * 3 * C);
    nntrainer::gemm_q4_0<float>(rows, 3 * C, K, x_fq.data(), K,
                                in_w.getData<char>(), 3 * C, p_fq.data(),
                                3 * C);
    std::vector<float> g_fq(static_cast<size_t>(rows) * C),
      y_fq(static_cast<size_t>(rows) * C);
    for (size_t r = 0; r < rows; ++r) {
      const float *a = p_fq.data() + r * 3 * C;
      for (unsigned int j = 0; j < C; ++j)
        g_fq[r * C + j] = a[j] * a[2 * C + j];
    }
    nntrainer::causal_depthwise_conv1d_k3(g_fq.data(), w_ptr, nullptr,
                                          y_fq.data(), 1, rows, C);
    for (size_t r = 0; r < rows; ++r) {
      const float *b = p_fq.data() + r * 3 * C + C;
      for (unsigned int j = 0; j < C; ++j)
        y_fq[r * C + j] = b[j] * y_fq[r * C + j];
    }
    const double y_ratio = outlierRatio(y_fq.data(), rows, C);
    const double g_snr = snrDb(g, g_fq.data(), g_fq.size());
    fakeQuantRowsU8(y_fq.data(), rows, C);
    std::vector<float> o_fq(static_cast<size_t>(rows) * N);
    nntrainer::gemm_q4_0<float>(rows, N, C, y_fq.data(), C,
                                out_w.getData<char>(), N, o_fq.data(), N);
    std::fprintf(stderr,
                 "[conv_block] %s rows=%u  SNR out %.1f dB  state %.1f dB | "
                 "act-u8 only: out %.1f g %.1f | outlier max/rms x %.1f "
                 "y %.1f%s\n",
                 context.getName().c_str(), rows,
                 snrDb(out_step.getData<float>(), h_out.data(), h_out.size()),
                 snrDb(state, h_state.data(), h_state.size()),
                 snrDb(out_step.getData<float>(), o_fq.data(), o_fq.size()),
                 g_snr, x_ratio, y_ratio,
                 convBlockShadowEnabled() ? "  (shadow: CPU values used)" : "");

    // The weight recipes, emulated in f32 (see the helpers above). The
    // accelerator's own row is recipe 0 done for real; if the emulation
    // of it lands near the measured number the others are trustworthy.
    {
      std::vector<float> w_in_f(static_cast<size_t>(3) * C * K);
      std::vector<float> w_out_f(static_cast<size_t>(N) * C);
      nntrainer::Q4_0Utils::dequantizeQ4_0x4(
        in_w.getData<char>(), static_cast<int>(3 * C), static_cast<int>(K),
        w_in_f.data());
      nntrainer::Q4_0Utils::dequantizeQ4_0x4(
        out_w.getData<char>(), static_cast<int>(N), static_cast<int>(C),
        w_out_f.data());
      static const char *const names[] = {
        "cur int4/ch", "blk32(=Q4_0)", "blk64", "blk128", "blk256", "int8/ch"};
      std::string line = "[conv_block]   recipes (f32 emulation, act u8):";
      for (int rcp = 0; rcp < 6; ++rcp) {
        std::vector<float> xr(in_step.getData<float>(),
                              in_step.getData<float>() +
                                static_cast<size_t>(rows) * K);
        std::vector<float> win(w_in_f), wout(w_out_f);
        auto quant_w = [&](float *w, unsigned int n_out, unsigned int k_in,
                           float *act, unsigned int act_rows) {
          (void)act;
          (void)act_rows;
          if (rcp >= 1 && rcp <= 4)
            quantChannelsBlocked(w, n_out, k_in, 32u << (rcp - 1));
          else if (rcp == 5)
            quantChannelsSym(w, n_out, k_in, 127L);
          else
            quantChannelsCur(w, n_out, k_in);
        };
        quant_w(win.data(), 3 * C, K, xr.data(), rows);
        fakeQuantRowsU8(xr.data(), rows, K);
        std::vector<float> pr(static_cast<size_t>(rows) * 3 * C);
        emuGemm(xr.data(), rows, K, win.data(), 3 * C, pr.data());
        std::vector<float> gr(static_cast<size_t>(rows) * C),
          yr(static_cast<size_t>(rows) * C);
        for (size_t r = 0; r < rows; ++r) {
          const float *a = pr.data() + r * 3 * C;
          for (unsigned int j = 0; j < C; ++j)
            gr[r * C + j] = a[j] * a[2 * C + j];
        }
        nntrainer::causal_depthwise_conv1d_k3(gr.data(), w_ptr, nullptr,
                                              yr.data(), 1, rows, C);
        for (size_t r = 0; r < rows; ++r) {
          const float *b = pr.data() + r * 3 * C + C;
          for (unsigned int j = 0; j < C; ++j)
            yr[r * C + j] = b[j] * yr[r * C + j];
        }
        quant_w(wout.data(), N, C, yr.data(), rows);
        fakeQuantRowsU8(yr.data(), rows, C);
        std::vector<float> orow(static_cast<size_t>(rows) * N);
        emuGemm(yr.data(), rows, C, wout.data(), N, orow.data());
        char buf[96];
        std::snprintf(
          buf, sizeof(buf), "  %s out %.1f g %.1f |", names[rcp],
          snrDb(out_step.getData<float>(), orow.data(), orow.size()),
          snrDb(g, gr.data(), gr.size()));
        line += buf;
      }
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    if (!convBlockShadowEnabled()) {
      std::memcpy(out_step.getData<float>(), h_out.data(),
                  h_out.size() * sizeof(float));
      std::memcpy(state, h_state.data(), h_state.size() * sizeof(float));
    }
  }
}

void ConvBlockLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();
  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());
  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

} // namespace causallm
