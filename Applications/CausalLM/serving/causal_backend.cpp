// SPDX-License-Identifier: Apache-2.0
#include "causal_backend.h"

namespace causallm::serving {
CausalBackend::CausalBackend(CausalLM &model_, const ChatTemplate *template_) :
  model(model_), chat_template(template_) {}
std::string CausalBackend::renderChat(const Json &request) {
  if (!chat_template)
    throw ApiError(400, "Model has no chat template; use /v1/completions",
                   "chat_template_missing");
  try {
    // Render the entire OpenAI history, never append a second copy of it.
    Json input = {{"messages", request.at("messages")}};
    if (request.contains("chat_template_kwargs"))
      input.update(request["chat_template_kwargs"]);
    return chat_template->apply(input);
  } catch (const std::exception &e) {
    throw ApiError(400, e.what(), "invalid_messages");
  }
}
uint64_t CausalBackend::create(const std::string &prompt) {
  const auto count = model.getTokenizer()->Encode(prompt, true).size();
  if (!count || count >= model.getMaxSeqLen())
    throw ApiError(400, "Prompt exceeds context window",
                   "context_length_exceeded");
  model.prepareKVSessionStorage(static_cast<unsigned int>(count));
  return model.createKVSession();
}
void CausalBackend::destroy(uint64_t cache) {
  model.selectKVSession(0);
  model.destroyKVSession(cache);
}
Generated CausalBackend::generate(uint64_t cache, const Generation &request) {
  const auto count = model.getTokenizer()->Encode(request.prompt, true).size();
  const auto limit = model.getMaxSeqLen();
  if (!count || count > model.getPrefillTokenLimit() || count >= limit ||
      request.max_tokens > limit - count)
    throw ApiError(400, "Prompt and max_tokens exceed context/prefill window",
                   "context_length_exceeded");
  const auto previous_count = model.getNumToGenerate();
  const auto previous_sampling = model.getSamplingParams();
  struct Restore {
    CausalLM &model;
    unsigned int count;
    CausalLM::SamplingParams sampling;
    ~Restore() {
      model.setNumToGenerate(count);
      model.setSamplingParams(sampling);
    }
  } restore{model, previous_count, previous_sampling};
  model.selectKVSession(cache);
  // OpenAI callers send complete histories, including edited branches. Prefix
  // reuse compares token IDs in the KV manager; a session id never proves a
  // hit.
  model.setSessionTokenPosition(0);
  model.setNumToGenerate(request.max_tokens);
  model.setSamplingParams({request.temperature, 0, request.top_p});
  model.setSamplingSeed(request.seed.value_or(model.getSamplingSeed()));
  model.prepareForRun();
  try {
    model.run(request.prompt.c_str(), request.temperature > 0, "", "", false);
    const auto metrics = model.getPerformanceMetrics();
    if (metrics.prompt_truncated_tokens)
      throw std::runtime_error("Inference unexpectedly truncated the prompt");
    auto text = model.getOutput();
    if (model.lastGenerationReachedEOS()) {
      for (auto id : model.getEosTokenIds()) {
        const auto marker =
          model.getTokenizer()->Decode({static_cast<int32_t>(id)});
        if (!marker.empty() && text.size() >= marker.size() &&
            text.compare(text.size() - marker.size(), marker.size(), marker) ==
              0) {
          text.resize(text.size() - marker.size());
          break;
        }
      }
    }
    return {text, count, metrics.generation_tokens,
            model.getLastPrefixReusedTokens(),
            model.lastGenerationReachedEOS() ? "stop" : "length"};
  } catch (const std::exception &e) {
    throw ApiError(500, e.what(), "inference_failed");
  }
}
} // namespace causallm::serving
