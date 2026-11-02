// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP compute Context implementation.
 */

#ifdef ENABLE_HEXKL

#include <htp_context.h>

#include <app_context.h>
#include <compute_ops.h>
#include <context_data.h>
#include <htp_backend.h>
#include <htp_decode_runtime.h>
#include <neuralnet.h>
#include <nntr_htp_decode_hook.h>
#include <tensor.h>

namespace nntrainer {

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const std::string &type,
                              const std::vector<std::string> &properties) {
  // HTP has no custom layers; reuse the CPU app context's factories.
  return AppContext::Global().createLayerObject(type, properties);
}

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const int int_key,
                              const std::vector<std::string> &properties) {
  return AppContext::Global().createLayerObject(int_key, properties);
}

sharedConstTensors HtpContext::runDecode(NeuralNetwork &nn, unsigned int from,
                                         unsigned int to,
                                         const sharedConstTensors &input,
                                         const sharedConstTensors &label) {
  // Keep initial prefill and single-row decode on their original hot paths.
  auto *runtime = to > from && to - from > 1u && htpDecodeActive()
                    ? getHtpDecodeRuntime()
                    : nullptr;
  if (!runtime || !runtime->canContinue(from) || !label.empty())
    return Context::runDecode(nn, from, to, input, label);

  // The supported causal graph has one token-ID vector and one vocabulary row.
  // Cache inputs retain their original MemoryData and are never reseeded here.
  size_t token_input = input.size();
  for (size_t i = 0; i < input.size(); ++i) {
    const auto &t = *input[i];
    if (t.getDataType() == Tdatatype::FP32 && t.batch() == 1u &&
        t.channel() == 1u && t.height() == 1u && t.width() >= to - from) {
      if (token_input != input.size())
        return Context::runDecode(nn, from, to, input, label);
      token_input = i;
    }
  }
  if (token_input == input.size())
    return Context::runDecode(nn, from, to, input, label);
  runtime->checkContinuationRange(from, to);

  auto token = std::make_shared<Tensor>(input[token_input]->getDim(), true);
  auto step_input = input;
  step_input[token_input] = token;
  const float *ids = input[token_input]->getData<float>();
  HtpDecodeRuntime::LogitsRequest logits(*runtime);
  sharedConstTensors result;
  for (unsigned pos = from; pos < to; ++pos) {
    token->getData<float>()[0] = ids[pos - from];
    // Intermediate rows need no vocabulary copy. The last row fulfils the
    // tensor-output contract of multi-token prefill, including sampling.
    logits.set(pos + 1u == to);
    result = nn.incremental_forwarding(pos, pos + 1u, step_input, label, false);
    if (result.size() != 1u || result[0]->size() != runtime->graph_words_[5] ||
        result[0]->getDataType() != Tdatatype::FP32)
      throw std::logic_error(
        "HTP continuation: expected one FP32 vocabulary row");
    if (runtime->have_id_) {
      unsigned int unused;
      get_htp_ops()->take_decode_token_id(&unused);
    }
  }
  return result;
}

void HtpContext::initialize() noexcept {
  // Ensure the CPU backend ops table exists — it is both the fallback
  // and the base class of HtpComputeOps. Go through ensureComputeOps()
  // rather than init_backend() directly: init_backend() is not
  // idempotent (__ggml_init, __openblas_set_num_threads), and
  // ensureComputeOps() is the call_once funnel every other Context
  // routes through.
  ensureComputeOps();

  auto cd = getContextData();
  if (!cd)
    return;

  // Bind the ops table lazily. A CPU-only API handle must not open the DSP
  // before a later HTP load supplies its library directory and fixed options.
  // The accelerated operations themselves check HtpBackend::enabled().
  try {
    cd->setComputeOps(get_htp_ops());
  } catch (...) {
    cd->setComputeOps(get_cpu_ops());
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
