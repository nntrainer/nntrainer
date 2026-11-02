// SPDX-License-Identifier: Apache-2.0
#include "causal_backend.h"
#include "api/streamer.h"
#include <algorithm>

namespace causallm::serving {
namespace {
// Hold only suffixes that could become an EOS marker. This handles markers
// split across callbacks without delaying ordinary text or splitting UTF-8.
struct DecodedStream {
  const Generation &request;
  std::vector<std::string> markers;
  std::string pending;
  std::exception_ptr failure;
  bool send(size_t count) {
    if (request.cancelled && request.cancelled())
      return false;
    if (!count)
      return true;
    auto text = pending.substr(0, count);
    pending.erase(0, count);
    return request.on_delta(text);
  }
  int put(const char *delta) noexcept {
    try {
      pending += delta;
      size_t keep = 0;
      for (const auto &marker : markers)
        for (size_t n = 1; n <= std::min(pending.size(), marker.size()); ++n)
          if (pending.compare(pending.size() - n, n, marker, 0, n) == 0)
            keep = std::max(keep, n);
      size_t count = pending.size() - keep;
      while (count && count < pending.size() &&
             (static_cast<unsigned char>(pending[count]) & 0xc0) == 0x80)
        --count;
      return send(count) ? 0 : 1;
    } catch (...) {
      failure = std::current_exception();
      return 1;
    }
  }
  bool flush(bool eos) {
    if (failure)
      std::rethrow_exception(failure);
    if (eos)
      for (const auto &marker : markers)
        if (!marker.empty() && pending.size() >= marker.size() &&
            pending.compare(pending.size() - marker.size(), marker.size(),
                            marker) == 0) {
          pending.resize(pending.size() - marker.size());
          break;
        }
    return send(pending.size());
  }
};
struct StreamBridge {
  BaseStreamer base;
  DecodedStream *state;
};
const BaseStreamerVTable stream_vtable = {
  [](BaseStreamer *base, const char *text) {
    return reinterpret_cast<StreamBridge *>(base)->state->put(text);
  },
  [](BaseStreamer *) {}};
} // namespace

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
  DecodedStream decoded{request, {}, {}, nullptr};
  for (auto id : model.getEosTokenIds())
    decoded.markers.push_back(
      model.getTokenizer()->Decode({static_cast<int32_t>(id)}));
  StreamBridge bridge{{&stream_vtable}, &decoded};
  struct Detach {
    CausalLM &model;
    ~Detach() { model.setStreamer(nullptr); }
  } detach{model};
  if (request.on_delta)
    model.setStreamer(&bridge.base);
  if (request.cancelled && request.cancelled())
    throw ApiError(499, "Request cancelled", "request_cancelled");
  if (request.on_start)
    request.on_start();
  try {
    model.run(request.prompt.c_str(), request.temperature > 0, "", "", false);
    if ((request.cancelled && request.cancelled()) ||
        (request.on_delta && !decoded.flush(model.lastGenerationReachedEOS())))
      throw ApiError(499, "Generation cancelled", "request_cancelled");
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
  } catch (const ApiError &) {
    throw;
  } catch (const std::exception &e) {
    throw ApiError(500, e.what(), "inference_failed");
  }
}
} // namespace causallm::serving
