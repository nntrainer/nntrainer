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
                                  Json body) {
  std::packaged_task<Json()> task(
    [this, method = std::move(method), path = std::move(path),
     body = std::move(body)] { return dispatch(method, path, body); });
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
    {
      std::unique_lock<std::mutex> lock(mutex);
      available.wait(lock, [this] { return stopping || !queue.empty(); });
      if (queue.empty())
        break;
      task = std::move(queue.front());
      queue.pop_front();
    }
    task(); // packaged_task transports exceptions to the HTTP caller.
  }
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
      if (!it->second.pinned && (victim == sessions.end() ||
                                 it->second.touched < victim->second.touched))
        victim = it;
    if (victim == sessions.end())
      throw ApiError(503, "All sessions are pinned; delete a session",
                     "session_capacity");
    eraseSession(victim->first);
  }
  auto id = "session-" + std::to_string(++sequence);
  sessions.emplace(id, Session{std::nullopt, sequence, pinned});
  return id;
}
void Service::eraseSession(const std::string &id) {
  auto it = sessions.find(id);
  if (it == sessions.end())
    throw ApiError(404, "Unknown session", "session_not_found");
  if (it->second.cache)
    backend.destroy(*it->second.cache);
  sessions.erase(it);
}
Json Service::dispatch(const std::string &method, const std::string &path,
                       const Json &body) {
  if (method == "POST" && path == "/sessions")
    return {{"id", createSession(true)}, {"object", "session"}};
  if (method == "GET" && path == "/sessions") {
    auto data = Json::array();
    for (const auto &entry : sessions)
      data.push_back({{"id", entry.first},
                      {"object", "session"},
                      {"pinned", entry.second.pinned}});
    return {{"object", "list"}, {"data", data}};
  }
  if (method == "DELETE" && path.rfind("/sessions/", 0) == 0) {
    auto id = path.substr(10);
    eraseSession(id);
    return {{"id", id}, {"deleted", true}, {"object", "session"}};
  }
  if (method == "POST" &&
      (path == "/v1/chat/completions" || path == "/v1/completions"))
    return completion(body, path == "/v1/chat/completions");
  throw ApiError(404, "Unknown route", "not_found");
}
Json Service::completion(const Json &body, bool chat) {
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
                                                  "chat_template_kwargs"};
  for (auto it = body.begin(); it != body.end(); ++it)
    require(supported.count(it.key()), "Unsupported parameter: " + it.key());
  require(body.contains("model") && body["model"].is_string(),
          "model is required");
  if (body["model"] != model)
    throw ApiError(404, "Unknown model", "model_not_found");
  if (body.contains("stream"))
    require(body["stream"].is_boolean() && !body["stream"].get<bool>(),
            "Only stream=false is supported");
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
  std::string id;
  if (body.contains("session_id")) {
    require(body["session_id"].is_string(), "session_id must be a string");
    id = body["session_id"].get<std::string>();
    if (!sessions.count(id))
      throw ApiError(404, "Unknown session", "session_not_found");
  } else {
    id = createSession(false);
  }
  auto &session = sessions.at(id);
  session.touched = ++sequence;
  Generated result;
  try {
    if (!session.cache)
      session.cache = backend.create(request.prompt);
    result = backend.generate(*session.cache, request);
  } catch (const ApiError &e) {
    // Validation errors occur before inference and preserve pinned history.
    if (e.status >= 500 || !body.contains("session_id")) {
      try {
        eraseSession(id);
      } catch (...) {
      }
    }
    throw;
  } catch (...) {
    // Never resume cache rows left by a failed inference.
    auto original = std::current_exception();
    try {
      eraseSession(id);
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
  return {
    {"id", (chat ? "chatcmpl-" : "cmpl-") + std::to_string(++sequence)},
    {"object", chat ? "chat.completion" : "text_completion"},
    {"created", std::time(nullptr)},
    {"model", model},
    {"session_id", id},
    {"choices", Json::array({choice})},
    {"usage",
     {{"prompt_tokens", result.prompt_tokens},
      {"completion_tokens", result.completion_tokens},
      {"total_tokens", result.prompt_tokens + result.completion_tokens},
      {"prompt_tokens_details", {{"cached_tokens", result.cached_tokens}}}}}};
}
} // namespace causallm::serving
