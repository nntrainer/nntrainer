// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "json.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace causallm::serving {
using Json = nlohmann::json;

/** API error with an HTTP status and OpenAI error code. */
struct ApiError : std::runtime_error {
  int status;
  std::string code;
  ApiError(int status_, const std::string &message,
           std::string code_ = "invalid_request") :
    std::runtime_error(message), status(status_), code(std::move(code_)) {}
};

/** Per-request bounded SSE mailbox. Producers never wait for a slow client.
 * Overflow and disconnect set cancellation; only the inference owner releases
 * the affected KV session. Errors before start remain ordinary HTTP errors.
 */
class EventStream {
public:
  explicit EventStream(size_t max_events = 128, size_t max_bytes = 256 * 1024);
  void start();
  void waitStarted(const std::function<bool()> &disconnected = {});
  bool push(const Json &event);
  void finish(std::exception_ptr error = nullptr);
  void cancel();
  bool cancelled() const;
  /** 1 = frame, 0 = timeout, -1 = end. Timeout allows socket checks. */
  int next(std::string &frame);

private:
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::string> events;
  const size_t max_events, max_bytes;
  size_t bytes = 0;
  bool started = false, done = false, aborted = false, terminal_sent = false;
  std::exception_ptr error;
};

/** One generation's validated parameters; prompt is the complete history. */
struct Generation {
  std::string prompt;
  unsigned int max_tokens = 16;
  float temperature = 1.0f;
  float top_p = 1.0f;
  std::optional<uint64_t> seed;
  // Backend calls after validation, before any model output. A false delta
  // return requests cooperative cancellation; callbacks belong to this request.
  std::function<void()> on_start;
  std::function<bool(const std::string &)> on_delta;
  std::function<bool()> cancelled;
};
struct Generated {
  std::string text;
  size_t prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
  std::string finish_reason = "stop";
};

/** Inference boundary. Only the service's worker may call these methods. */
class Backend {
public:
  virtual ~Backend() = default;
  virtual std::string renderChat(const Json &request) = 0;
  virtual uint64_t create(const std::string &prompt) = 0;
  virtual void destroy(uint64_t cache) = 0;
  virtual Generated generate(uint64_t cache, const Generation &request) = 0;
};

/** Bounded FIFO feeding a single model owner. HTTP threads never access KV.
 * An active generation is allowed to finish during shutdown, then pending
 * requests drain and session storage is released before the worker exits.
 */
class Service {
public:
  Service(Backend &backend, std::string model, size_t max_sessions = 32,
          size_t queue_capacity = 32);
  ~Service();
  Service(const Service &) = delete;
  Service &operator=(const Service &) = delete;
  /** Enqueue API work or throw 503 when the waiting queue is full. */
  std::future<Json> submit(std::string method, std::string path, Json body,
                           std::shared_ptr<EventStream> stream = nullptr);
  const std::string &modelId() const { return model; }

private:
  struct Session {
    std::optional<uint64_t> cache;
    uint64_t touched = 0;
    bool pinned = false;
    bool busy = false;
  };
  Backend &backend;
  const std::string model;
  const size_t max_sessions, queue_capacity;
  std::map<std::string, Session> sessions;
  uint64_t sequence = 0;
  std::mutex mutex;
  std::condition_variable available;
  std::deque<std::packaged_task<Json()>> queue;
  std::deque<uint64_t> retired; // Released only by the model worker.
  bool stopping = false;
  std::thread worker;
  void loop();
  Json dispatch(const std::string &method, const std::string &path,
                const Json &body, const std::shared_ptr<EventStream> &stream);
  std::string createSession(bool pinned);
  void eraseSession(const std::string &id, bool allow_busy = false);
  Json control(const std::string &method, const std::string &path);
  Json completion(const Json &body, bool chat,
                  const std::shared_ptr<EventStream> &stream);
};

/** OpenAI-shaped error payload, including for transport errors. */
Json errorBody(int status, const std::string &message, const std::string &code);
} // namespace causallm::serving
