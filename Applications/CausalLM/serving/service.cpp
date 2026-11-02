// SPDX-License-Identifier: Apache-2.0
#include "service.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <set>

namespace causallm::serving {
namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw ApiError(400, message);
}
unsigned int positive(const Json &value, const std::string &key) {
  require(value.is_number_integer() && value >= 1 && value <= 2147483647,
          key + " must be a positive integer");
  return value.get<unsigned int>();
}
float bounded(const Json &value, float lo, float hi, const std::string &key) {
  require(value.is_number(), key + " must be a number");
  auto x = value.get<double>();
  require(std::isfinite(x) && x >= lo && x <= hi, key + " is out of range");
  return static_cast<float>(x);
}
} // namespace

Json errorBody(int status, const std::string &message,
               const std::string &code) {
  return {{"error",
           {{"message", message},
            {"type", status >= 500 ? "server_error" : "invalid_request_error"},
            {"param", nullptr},
            {"code", code}}}};
}

EventStream::EventStream(size_t max_events_, size_t max_bytes_) :
  max_events(max_events_), max_bytes(max_bytes_) {
  if (!max_events || !max_bytes)
    throw std::invalid_argument("Stream bounds must be positive");
}
void EventStream::start() {
  std::lock_guard<std::mutex> lock(mutex);
  if (aborted)
    throw ApiError(499, "Request cancelled", "request_cancelled");
  started = true;
  changed.notify_all();
}
void EventStream::waitStarted(const std::function<bool()> &disconnected) {
  std::unique_lock<std::mutex> lock(mutex);
  while (!started && !done && !aborted) {
    changed.wait_for(lock, std::chrono::milliseconds(100));
    if (disconnected && disconnected()) {
      aborted = true;
      changed.notify_all();
    }
  }
  if (!started) {
    if (error)
      std::rethrow_exception(error);
    throw ApiError(499, "Request cancelled", "request_cancelled");
  }
}
bool EventStream::push(const Json &event) {
  auto frame = "data: " + event.dump() + "\n\n";
  std::lock_guard<std::mutex> lock(mutex);
  if (aborted || done)
    return false;
  if (events.size() >= max_events || frame.size() > max_bytes - bytes) {
    aborted = true;
    error = std::make_exception_ptr(
      ApiError(503, "Streaming client is too slow", "stream_overflow"));
    changed.notify_all();
    return false;
  }
  bytes += frame.size();
  events.push_back(std::move(frame));
  changed.notify_all();
  return true;
}
void EventStream::finish(std::exception_ptr failure) {
  std::lock_guard<std::mutex> lock(mutex);
  if (done)
    return;
  if (!error)
    error = failure;
  done = true;
  changed.notify_all();
}
void EventStream::cancel() {
  std::lock_guard<std::mutex> lock(mutex);
  aborted = true;
  changed.notify_all();
}
bool EventStream::cancelled() const {
  std::lock_guard<std::mutex> lock(mutex);
  return aborted;
}
int EventStream::next(std::string &frame) {
  std::unique_lock<std::mutex> lock(mutex);
  if (!changed.wait_for(lock, std::chrono::milliseconds(1000),
                        [this] { return !events.empty() || done; }))
    return 0;
  if (!events.empty()) {
    frame = std::move(events.front());
    bytes -= frame.size();
    events.pop_front();
    return 1;
  }
  if (terminal_sent)
    return -1;
  terminal_sent = true;
  frame.clear();
  if (error) {
    try {
      std::rethrow_exception(error);
    } catch (const ApiError &e) {
      frame = "data: " + errorBody(e.status, e.what(), e.code).dump() + "\n\n";
    } catch (...) {
      frame =
        "data: " +
        errorBody(500, "Inference service failed", "internal_error").dump() +
        "\n\n";
    }
  }
  frame += "data: [DONE]\n\n";
  return 1;
}

Service::Service(Backend &backend_, std::string model_, size_t max_sessions_,
                 size_t queue_capacity_) :
  backend(backend_),
  model(std::move(model_)),
  max_sessions(max_sessions_),
  queue_capacity(queue_capacity_) {
  if (!max_sessions || !queue_capacity)
    throw std::invalid_argument(
      "Session and queue capacities must be positive");
  worker = std::thread(&Service::loop, this);
}
Service::~Service() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    stopping = true;
  }
  available.notify_one();
  worker.join();
}
std::future<Json> Service::submit(std::string method, std::string path,
                                  Json body,
                                  std::shared_ptr<EventStream> stream) {
  if (((method == "GET" || method == "POST") && path == "/sessions") ||
      (method == "DELETE" && path.rfind("/sessions/", 0) == 0)) {
    std::promise<Json> response;
    auto result = response.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex);
      try {
        if (stopping)
          throw ApiError(503, "Service is shutting down", "server_busy");
        response.set_value(control(method, path));
      } catch (...) {
        response.set_exception(std::current_exception());
      }
    }
    available.notify_one();
    return result;
  }
  std::packaged_task<Json()> task([this, method = std::move(method),
                                   path = std::move(path),
                                   body = std::move(body), stream] {
    try {
      if (stream && stream->cancelled())
        throw ApiError(499, "Request cancelled", "request_cancelled");
      auto result = dispatch(method, path, body, stream);
      if (stream)
        stream->finish();
      return result;
    } catch (...) {
      if (stream)
        stream->finish(std::current_exception());
      throw;
    }
  });
  auto result = task.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping || queue.size() >= queue_capacity)
      throw ApiError(503, "Inference queue is full or shutting down",
                     "server_busy");
    queue.push_back(std::move(task));
  }
  available.notify_one();
  return result;
}
void Service::loop() {
  for (;;) {
    std::packaged_task<Json()> task;
    std::deque<uint64_t> reclaim;
    {
      std::unique_lock<std::mutex> lock(mutex);
      available.wait(lock, [this] {
        return stopping || !queue.empty() || !retired.empty();
      });
      reclaim.swap(retired);
      if (!queue.empty()) {
        task = std::move(queue.front());
        queue.pop_front();
      } else if (stopping && reclaim.empty())
        break;
    }
    for (auto cache : reclaim)
      try {
        backend.destroy(cache);
      } catch (...) {
      }
    if (task.valid())
      task();
  }
  // Admission has stopped and queued work is drained; only this worker owns KV.
  for (auto &entry : sessions)
    if (entry.second.cache)
      try {
        backend.destroy(*entry.second.cache);
      } catch (...) {
      }
}
std::string Service::createSession(bool pinned) {
  if (sessions.size() == max_sessions) {
    auto victim = sessions.end();
    for (auto it = sessions.begin(); it != sessions.end(); ++it)
      if (!it->second.pinned && !it->second.busy &&
          (victim == sessions.end() ||
           it->second.touched < victim->second.touched))
        victim = it;
    if (victim == sessions.end())
      throw ApiError(
        503, "All sessions are pinned or active; delete an idle session",
        "session_capacity");
    eraseSession(victim->first);
  }
  auto id = "session-" + std::to_string(++sequence);
  sessions.emplace(id, Session{std::nullopt, sequence, pinned});
  return id;
}
void Service::eraseSession(const std::string &id, bool allow_busy) {
  auto it = sessions.find(id);
  if (it == sessions.end())
    throw ApiError(404, "Unknown session", "session_not_found");
  if (it->second.busy && !allow_busy)
    throw ApiError(409,
                   "Session is generating; retry deletion after it finishes",
                   "session_busy");
  if (it->second.cache)
    retired.push_back(*it->second.cache);
  sessions.erase(it);
}
// The caller holds mutex. No control operation calls the model or waits on
// inference.
Json Service::control(const std::string &method, const std::string &path) {
  if (method == "POST" && path == "/sessions")
    return {{"id", createSession(true)}, {"object", "session"}};
  if (method == "GET" && path == "/sessions") {
    auto data = Json::array();
    for (const auto &entry : sessions)
      data.push_back({{"id", entry.first},
                      {"object", "session"},
                      {"pinned", entry.second.pinned},
                      {"state", entry.second.busy ? "generating" : "idle"}});
    return {{"object", "list"}, {"data", data}};
  }
  if (method == "DELETE" && path.rfind("/sessions/", 0) == 0) {
    auto id = path.substr(10);
    eraseSession(id);
    return {{"id", id}, {"deleted", true}, {"object", "session"}};
  }
  throw ApiError(404, "Unknown route", "not_found");
}
Json Service::dispatch(const std::string &method, const std::string &path,
                       const Json &body,
                       const std::shared_ptr<EventStream> &stream) {
  if (method == "POST" &&
      (path == "/v1/chat/completions" || path == "/v1/completions"))
    return completion(body, path == "/v1/chat/completions", stream);
  throw ApiError(404, "Unknown route", "not_found");
}
Json Service::completion(const Json &body, bool chat,
                         const std::shared_ptr<EventStream> &stream) {
  require(body.is_object(), "Expected a JSON object");
  static const std::set<std::string> supported = {"model",
                                                  "messages",
                                                  "prompt",
                                                  "stream",
                                                  "n",
                                                  "max_tokens",
                                                  "max_completion_tokens",
                                                  "temperature",
                                                  "top_p",
                                                  "seed",
                                                  "session_id",
                                                  "user",
                                                  "chat_template_kwargs",
                                                  "stream_options"};
  for (auto it = body.begin(); it != body.end(); ++it)
    require(supported.count(it.key()), "Unsupported parameter: " + it.key());
  require(body.contains("model") && body["model"].is_string(),
          "model is required");
  if (body["model"] != model)
    throw ApiError(404, "Unknown model", "model_not_found");
  if (body.contains("stream"))
    require(body["stream"].is_boolean(), "stream must be a boolean");
  require(body.value("stream", false) == static_cast<bool>(stream),
          "Streaming transport required for stream=true");
  bool include_usage = false;
  if (body.contains("stream_options")) {
    require(stream && body["stream_options"].is_object(),
            "stream_options requires stream=true");
    for (auto it = body["stream_options"].begin();
         it != body["stream_options"].end(); ++it)
      require(it.key() == "include_usage" && it.value().is_boolean(),
              "Only boolean stream_options.include_usage is supported");
    include_usage = body["stream_options"].value("include_usage", false);
  }
  if (body.contains("n"))
    require(body["n"].is_number_integer() && body["n"] == 1,
            "Only n=1 is supported");
  Generation request;
  require(
    !(body.contains("max_tokens") && body.contains("max_completion_tokens")),
    "Specify only one token limit");
  if (body.contains("max_tokens"))
    request.max_tokens = positive(body["max_tokens"], "max_tokens");
  if (body.contains("max_completion_tokens"))
    request.max_tokens =
      positive(body["max_completion_tokens"], "max_completion_tokens");
  if (body.contains("temperature"))
    request.temperature = bounded(body["temperature"], 0, 2, "temperature");
  if (body.contains("top_p"))
    request.top_p = bounded(body["top_p"], 0, 1, "top_p");
  if (body.contains("seed")) {
    require(body["seed"].is_number_integer(), "seed must be an integer");
    request.seed = body["seed"].get<uint64_t>();
  }
  if (body.contains("chat_template_kwargs")) {
    const auto &kwargs = body["chat_template_kwargs"];
    require(chat && kwargs.is_object(),
            "chat_template_kwargs must be a chat options object");
    for (auto it = kwargs.begin(); it != kwargs.end(); ++it)
      require(
        it.key() == "enable_thinking" && it.value().is_boolean(),
        "Only boolean enable_thinking is supported in chat_template_kwargs");
  }
  if (chat) {
    require(!body.contains("prompt"), "prompt is not a chat parameter");
    require(body.contains("messages") && body["messages"].is_array() &&
              !body["messages"].empty(),
            "messages must be a nonempty array");
    for (auto &message : body["messages"]) {
      require(message.is_object() && message.contains("role") &&
                message["role"].is_string() && message.contains("content") &&
                message["content"].is_string(),
              "Text messages require role and content strings");
      const auto role = message["role"].get<std::string>();
      require(role == "system" || role == "developer" || role == "user" ||
                role == "assistant",
              "Unsupported message role");
      for (auto it = message.begin(); it != message.end(); ++it)
        require(it.key() == "role" || it.key() == "content",
                "Unsupported message field: " + it.key());
    }
    request.prompt = backend.renderChat(body);
  } else {
    require(!body.contains("messages"),
            "messages is not a completion parameter");
    require(body.contains("prompt") && body["prompt"].is_string(),
            "prompt must be a string");
    request.prompt = body["prompt"].get<std::string>();
  }
  require(!request.prompt.empty(), "Prompt must not be empty");
  std::string id, completion_id;
  std::optional<uint64_t> cache;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (body.contains("session_id")) {
      require(body["session_id"].is_string(), "session_id must be a string");
      id = body["session_id"].get<std::string>();
      if (!sessions.count(id))
        throw ApiError(404, "Unknown session", "session_not_found");
    } else
      id = createSession(false);
    auto &session = sessions.at(id);
    session.touched = ++sequence;
    session.busy = true;
    cache = session.cache;
    completion_id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(++sequence);
  }
  struct BusyGuard {
    Service &service;
    const std::string &id;
    ~BusyGuard() {
      std::lock_guard<std::mutex> lock(service.mutex);
      auto it = service.sessions.find(id);
      if (it != service.sessions.end())
        it->second.busy = false;
    }
  } busy{*this, id};
  const auto created = std::time(nullptr);
  Json prototype = {
    {"id", completion_id},
    {"object", chat ? "chat.completion.chunk" : "text_completion"},
    {"created", created},
    {"model", model},
    {"session_id", id}};
  if (include_usage)
    prototype["usage"] = nullptr;
  auto chunk = [prototype, chat](const std::string &text, bool initial,
                                 const Json &reason) mutable {
    Json choice = {
      {"index", 0}, {"finish_reason", reason}, {"logprobs", nullptr}};
    if (chat) {
      choice["delta"] = Json::object();
      if (initial)
        choice["delta"] = {{"role", "assistant"}, {"content", ""}};
      else if (!text.empty())
        choice["delta"]["content"] = text;
    } else
      choice["text"] = text;
    prototype["choices"] = Json::array({choice});
    return prototype;
  };
  if (stream) {
    request.cancelled = [stream] { return stream->cancelled(); };
    request.on_start = [stream, chunk]() mutable {
      stream->start();
      if (!stream->push(chunk("", true, nullptr)))
        throw ApiError(499, "Stream cancelled", "request_cancelled");
    };
    request.on_delta = [stream, chunk](const std::string &text) mutable {
      return !stream->cancelled() && stream->push(chunk(text, false, nullptr));
    };
  }
  Generated result;
  try {
    // Reclaim LRU victims before this request allocates new KV pages. Controls
    // may retire idle sessions concurrently but never touch native KV storage.
    std::deque<uint64_t> reclaim;
    {
      std::lock_guard<std::mutex> lock(mutex);
      reclaim.swap(retired);
    }
    for (auto old : reclaim)
      backend.destroy(old);
    if (!cache) {
      cache = backend.create(request.prompt);
      std::lock_guard<std::mutex> lock(mutex);
      sessions.at(id).cache = cache;
    }
    result = backend.generate(*cache, request);
    if (stream && stream->cancelled())
      throw ApiError(499, "Generation cancelled; session released",
                     "request_cancelled");
  } catch (const ApiError &e) {
    // Validation errors occur before inference and preserve pinned history.
    if (e.status >= 500 || (stream && stream->cancelled()) ||
        !body.contains("session_id")) {
      try {
        std::lock_guard<std::mutex> lock(mutex);
        eraseSession(id, true);
      } catch (...) {
      }
    }
    throw;
  } catch (...) {
    // Never resume cache rows left by a failed inference.
    auto original = std::current_exception();
    try {
      std::lock_guard<std::mutex> lock(mutex);
      eraseSession(id, true);
    } catch (...) {
    }
    std::rethrow_exception(original);
  }
  Json choice = {{"index", 0},
                 {"finish_reason", result.finish_reason},
                 {"logprobs", nullptr}};
  if (chat)
    choice["message"] = {{"role", "assistant"}, {"content", result.text}};
  else
    choice["text"] = result.text;
  Json response = {
    {"id", completion_id},
    {"object", chat ? "chat.completion" : "text_completion"},
    {"created", created},
    {"model", model},
    {"session_id", id},
    {"choices", Json::array({choice})},
    {"usage",
     {{"prompt_tokens", result.prompt_tokens},
      {"completion_tokens", result.completion_tokens},
      {"total_tokens", result.prompt_tokens + result.completion_tokens},
      {"prompt_tokens_details", {{"cached_tokens", result.cached_tokens}}}}}};
  if (stream) {
    bool accepted = stream->push(chunk("", false, result.finish_reason));
    if (accepted && include_usage) {
      prototype["choices"] = Json::array();
      prototype["usage"] = response["usage"];
      accepted = stream->push(prototype);
    }
    if (!accepted) {
      std::lock_guard<std::mutex> lock(mutex);
      eraseSession(id, true);
      throw ApiError(499, "Stream cancelled; session released",
                     "request_cancelled");
    }
  }
  return response;
}
} // namespace causallm::serving
