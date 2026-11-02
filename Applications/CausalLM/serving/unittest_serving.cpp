// SPDX-License-Identifier: Apache-2.0
#include "third_party/cpp-httplib/httplib.h"

#include "causal_backend.h"
#include "http_server.h"
#include "models/qwen2/qwen2_causallm.h"
#include "models/qwen3/qwen3_causallm.h"
#include "service.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <set>

using namespace causallm::serving;
namespace {
class FakeBackend : public Backend {
public:
  std::set<uint64_t> caches;
  uint64_t next = 0;
  bool fail = false;
  std::promise<void> entered, release;
  bool block = false;
  std::string renderChat(const Json &body) override {
    return body["messages"].dump();
  }
  uint64_t create(const std::string &) override {
    caches.insert(++next);
    return next;
  }
  void destroy(uint64_t id) override { caches.erase(id); }
  Generated generate(uint64_t id, const Generation &request) override {
    if (request.on_start)
      request.on_start();
    if (block) {
      entered.set_value();
      auto released = release.get_future();
      while (released.wait_for(std::chrono::milliseconds(20)) !=
             std::future_status::ready) {
        if (request.cancelled && request.cancelled()) {
          block = false;
          return {};
        }
      }
      block = false;
    }
    if (fail)
      throw ApiError(500, "failed", "inference_failed");
    if (!caches.count(id))
      throw std::logic_error("Wrong cache");
    if (request.on_delta)
      request.on_delta(request.prompt);
    return {request.prompt, 3, 1, 0, "length"};
  }
};
Json request() {
  return {{"model", "test"},
          {"prompt", "hello"},
          {"max_tokens", 1},
          {"temperature", 0}};
}
Json call(Service &service, Json body) {
  return service.submit("POST", "/v1/completions", std::move(body)).get();
}
// A worker task is a KV-reclamation barrier; session controls are intentionally
// immediate and no longer double as inference/reclamation barriers.
void settle(Service &service) {
  service.submit("GET", "/test-barrier", {}).wait();
}

std::string rawHttp(int port, const std::string &wire) {
  auto sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock == INVALID_SOCKET)
    throw std::runtime_error("socket failed");
  struct Close {
    socket_t sock;
    ~Close() { httplib::detail::close_socket(sock); }
  } close{sock};
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(sock, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
    throw std::runtime_error("connect failed");
  size_t offset = 0;
  while (offset < wire.size()) {
    auto sent = ::send(sock, wire.data() + offset,
                       static_cast<int>(wire.size() - offset), 0);
    if (sent <= 0)
      throw std::runtime_error("send failed");
    offset += static_cast<size_t>(sent);
  }
  std::string response;
  char buffer[4096];
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (httplib::detail::select_read(sock, 0, 100000) <= 0)
      continue;
    auto count = ::recv(sock, buffer, sizeof(buffer), 0);
    if (count <= 0)
      return response;
    response.append(buffer, static_cast<size_t>(count));
    auto end = response.find("\r\n\r\n");
    auto length = response.find("Content-Length: ");
    if (end != std::string::npos && length != std::string::npos &&
        length < end) {
      auto bytes = std::stoul(response.substr(length + 16));
      if (response.size() >= end + 4 + bytes)
        return response;
    }
  }
  throw std::runtime_error("Raw HTTP response timed out");
}
Json sseEvents(const std::string &wire) {
  Json events = Json::array();
  size_t from = 0;
  while (from < wire.size()) {
    auto end = wire.find("\n\n", from);
    if (end == std::string::npos)
      break;
    auto frame = wire.substr(from, end - from);
    from = end + 2;
    if (frame.rfind("data: ", 0) != 0)
      continue;
    auto data = frame.substr(6);
    if (data == "[DONE]")
      events.push_back("[DONE]");
    else
      events.push_back(Json::parse(data));
  }
  return events;
}
std::string drain(const std::shared_ptr<EventStream> &stream) {
  std::string wire, frame;
  for (int attempts = 0; attempts < 1000; ++attempts) {
    int state = stream->next(frame);
    if (state < 0)
      return wire;
    if (state > 0)
      wire += frame;
  }
  throw std::runtime_error("Stream did not terminate");
}
TEST(Serving, StreamingSchemasAndUsage) {
  for (bool chat : {false, true}) {
    FakeBackend backend;
    Service service(backend, "test");
    auto body = request();
    body["stream"] = true;
    body["stream_options"] = {{"include_usage", true}};
    auto path = chat ? "/v1/chat/completions" : "/v1/completions";
    if (chat) {
      body.erase("prompt");
      body["messages"] =
        Json::array({{{"role", "user"}, {"content", "hello"}}});
    }
    auto stream = std::make_shared<EventStream>();
    auto finished = service.submit("POST", path, body, stream);
    stream->waitStarted();
    auto events = sseEvents(drain(stream));
    auto final = finished.get();
    ASSERT_EQ(events.size(), 5u);
    EXPECT_EQ(events.back(), "[DONE]");
    EXPECT_EQ(events[2]["choices"][0]["finish_reason"], "length");
    EXPECT_TRUE(events[3]["choices"].empty());
    EXPECT_EQ(events[3]["usage"], final["usage"]);
    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(events[i]["id"], final["id"]);
    Json expected =
      chat ? Json{{"content", backend.renderChat(body)}} : Json("hello");
    EXPECT_EQ(events[1]["choices"][0][chat ? "delta" : "text"], expected);
    if (chat) {
      EXPECT_EQ(events[0]["choices"][0]["delta"]["role"], "assistant");
    }
  }
}
TEST(Serving, StreamingValidationBeforeStart) {
  FakeBackend backend;
  Service service(backend, "test");
  for (auto patch : {Json{{"model", "missing"}}, Json{{"max_tokens", 0}},
                     Json{{"stream_options", "invalid"}},
                     Json{{"stream_options", {{"include_usage", "true"}}}},
                     Json{{"stream_options", {{"unsupported", true}}}}}) {
    auto body = request();
    body.update(patch);
    body["stream"] = true;
    auto stream = std::make_shared<EventStream>();
    auto result = service.submit("POST", "/v1/completions", body, stream);
    EXPECT_THROW(stream->waitStarted(), ApiError);
    EXPECT_THROW(result.get(), ApiError);
  }
}
TEST(Serving, StreamBoundsAndOverflowCleanup) {
  for (auto bounds : {std::make_pair(1u, 1024u), std::make_pair(128u, 4u)}) {
    FakeBackend backend;
    Service service(backend, "test");
    auto body = request();
    body["stream"] = true;
    auto stream = std::make_shared<EventStream>(bounds.first, bounds.second);
    auto result = service.submit("POST", "/v1/completions", body, stream);
    EXPECT_THROW(result.get(), ApiError);
    auto wire = drain(stream);
    EXPECT_NE(wire.find("stream_overflow"), std::string::npos);
    EXPECT_NE(wire.find("[DONE]"), std::string::npos);
    EXPECT_TRUE(stream->cancelled());
    settle(service);
    EXPECT_TRUE(backend.caches.empty());
    EXPECT_TRUE(service.submit("GET", "/sessions", {}).get()["data"].empty());
  }
}
TEST(Serving, CancelActiveAndQueuedStreams) {
  FakeBackend backend;
  Service service(backend, "test");
  auto id = service.submit("POST", "/sessions", {}).get()["id"];
  auto queuedId = service.submit("POST", "/sessions", {}).get()["id"];
  auto body = request();
  body["session_id"] = id;
  body["stream"] = true;
  backend.block = true;
  auto entered = backend.entered.get_future();
  auto active = std::make_shared<EventStream>();
  auto running = service.submit("POST", "/v1/completions", body, active);
  entered.wait();
  body["session_id"] = queuedId;
  auto queued = std::make_shared<EventStream>();
  auto waiting = service.submit("POST", "/v1/completions", body, queued);
  queued->cancel();
  active->cancel();
  EXPECT_THROW(running.get(), ApiError);
  EXPECT_THROW(waiting.get(), ApiError);
  settle(service);
  EXPECT_TRUE(backend.caches.empty());
  auto retained = service.submit("GET", "/sessions", {}).get()["data"];
  ASSERT_EQ(retained.size(), 1u);
  EXPECT_EQ(retained[0]["id"],
            queuedId); // Never executed: preserve pinned KV owner.
  EXPECT_NO_THROW(call(service, request()));
}
TEST(Serving, StreamFailureAfterStart) {
  FakeBackend backend;
  backend.fail = true;
  Service service(backend, "test");
  auto body = request();
  body["stream"] = true;
  auto stream = std::make_shared<EventStream>();
  auto result = service.submit("POST", "/v1/completions", body, stream);
  stream->waitStarted();
  EXPECT_THROW(result.get(), ApiError);
  auto events = sseEvents(drain(stream));
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events[1]["error"]["code"], "inference_failed");
  EXPECT_EQ(events.back(), "[DONE]");
  settle(service);
  EXPECT_TRUE(backend.caches.empty());
}
TEST(Serving, SchemaAndChatHistory) {
  FakeBackend backend;
  Service service(backend, "test");
  auto result = call(service, request());
  EXPECT_EQ(result["object"], "text_completion");
  EXPECT_EQ(result["choices"][0]["text"], "hello");
  EXPECT_EQ(result["usage"]["total_tokens"], 4);
  Json chat = {
    {"model", "test"},
    {"messages", Json::array({{{"role", "user"}, {"content", "hello"}}})}};
  auto first = service.submit("POST", "/v1/chat/completions", chat).get();
  chat["session_id"] = first["session_id"];
  chat["messages"].push_back({{"role", "assistant"}, {"content", "answer"}});
  chat["messages"].push_back({{"role", "user"}, {"content", "next"}});
  auto second = service.submit("POST", "/v1/chat/completions", chat).get();
  EXPECT_EQ(second["object"], "chat.completion");
  EXPECT_EQ(second["choices"][0]["message"]["role"], "assistant");
  EXPECT_EQ(second["choices"][0]["message"]["content"],
            chat["messages"].dump());
}
TEST(Serving, InvalidRequests) {
  FakeBackend backend;
  Service service(backend, "test");
  for (auto patch :
       {Json{{"stream", true}}, Json{{"stream", "false"}}, Json{{"n", 2}},
        Json{{"max_tokens", -1}}, Json{{"max_tokens", 0}},
        Json{{"max_tokens", 1.5}}, Json{{"temperature", -1}},
        Json{{"temperature", "1"}}, Json{{"top_p", 2}},
        Json{{"model", "missing"}}, Json{{"prompt", Json::array()}},
        Json{{"stop", "end"}}, Json{{"session_id", "missing"}},
        Json{{"max_completion_tokens", 2}}}) {
    auto body = request();
    body.update(patch);
    EXPECT_THROW(call(service, body), ApiError) << body.dump();
  }
  EXPECT_THROW(call(service, Json::array()), ApiError);
}
TEST(Serving, InvalidMessages) {
  FakeBackend backend;
  Service service(backend, "test");
  for (auto messages : {Json::array(), Json("bad"),
                        Json::array({{{"role", "tool"}, {"content", "x"}}}),
                        Json::array({{{"role", "user"}, {"content", nullptr}}}),
                        Json::array({{{"role", "user"},
                                      {"content", "x"},
                                      {"tool_calls", Json::array()}}})}) {
    EXPECT_THROW(service
                   .submit("POST", "/v1/chat/completions",
                           {{"model", "test"}, {"messages", messages}})
                   .get(),
                 ApiError);
  }
}
TEST(Serving, SessionLruAndPinnedCapacity) {
  FakeBackend backend;
  {
    Service service(backend, "test", 2);
    auto pinned = service.submit("POST", "/sessions", {}).get()["id"];
    auto a = call(service, request());
    auto b = call(service, request());
    EXPECT_EQ(backend.caches.size(), 1u);
    EXPECT_NE(a["session_id"], b["session_id"]);
    auto body = request();
    body["session_id"] = pinned;
    call(service, body);
    auto pinned2 = service.submit("POST", "/sessions", {}).get()["id"];
    EXPECT_THROW(call(service, request()), ApiError);
    service.submit("DELETE", "/sessions/" + pinned.get<std::string>(), {})
      .get();
    settle(service);
    EXPECT_TRUE(backend.caches.empty());
    EXPECT_THROW(service.submit("DELETE", "/sessions/missing", {}).get(),
                 ApiError);
    EXPECT_FALSE(pinned2.empty());
  }
  EXPECT_TRUE(backend.caches.empty());
}
TEST(Serving, FailureReleasesSession) {
  FakeBackend backend;
  Service service(backend, "test");
  backend.fail = true;
  EXPECT_THROW(call(service, request()), ApiError);
  settle(service);
  EXPECT_TRUE(backend.caches.empty());
  EXPECT_TRUE(service.submit("GET", "/sessions", {}).get()["data"].empty());
  backend.fail = false;
  EXPECT_NO_THROW(call(service, request()));
}
TEST(Serving, BoundedQueueAndDrain) {
  FakeBackend backend;
  backend.block = true;
  auto entered = backend.entered.get_future();
  Service service(backend, "test", 4, 1);
  auto active = service.submit("POST", "/v1/completions", request());
  entered.wait();
  auto waiting = service.submit("POST", "/v1/completions", request());
  EXPECT_THROW(service.submit("POST", "/v1/completions", request()), ApiError);
  backend.release.set_value();
  EXPECT_EQ(active.get()["object"], "text_completion");
  EXPECT_EQ(waiting.get()["object"], "text_completion");
}
TEST(Serving, NativeHttp) {
  FakeBackend backend;
  Service service(backend, "test");
  HttpServer server(service);
  int port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0) << "Local socket permission is required";
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(5, 0);
  auto models = client.Get("/v1/models");
  ASSERT_TRUE(models);
  EXPECT_EQ(Json::parse(models->body)["data"][0]["id"], "test");
  auto good =
    client.Post("/v1/completions", request().dump(), "application/json");
  ASSERT_TRUE(good);
  EXPECT_EQ(good->status, 200);
  backend.block = true;
  auto body = request();
  body["stream"] = true;
  httplib::Request streamed;
  streamed.method = "POST";
  streamed.path = "/v1/completions";
  streamed.body = body.dump();
  streamed.set_header("Content-Type", "application/json");
  std::string wire;
  bool released = false;
  streamed.content_receiver = [&](const char *data, size_t length, uint64_t,
                                  uint64_t) {
    wire.append(data, length);
    if (!released) {
      released = true;
      backend.release.set_value();
    }
    return true;
  };
  auto live = client.send(streamed);
  if (!released)
    backend.release.set_value();
  ASSERT_TRUE(live);
  EXPECT_TRUE(released); // A chunk arrived while generation was still blocked.
  EXPECT_EQ(live->get_header_value("Content-Type"), "text/event-stream");
  EXPECT_EQ(sseEvents(wire).back(), "[DONE]");
  auto invalid = body;
  invalid["model"] = "missing";
  auto rejected =
    client.Post("/v1/completions", invalid.dump(), "application/json");
  ASSERT_TRUE(rejected);
  EXPECT_EQ(rejected->status, 404);
  EXPECT_EQ(rejected->get_header_value("Content-Type"), "application/json");
  auto bad = client.Post("/v1/chat/completions", "{", "application/json");
  ASSERT_TRUE(bad);
  EXPECT_EQ(bad->status, 400);
  EXPECT_TRUE(Json::parse(bad->body)["error"].contains("type"));
  auto unknown = client.Get("/missing");
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown->status, 404);
  EXPECT_TRUE(Json::parse(unknown->body).contains("error"));
}
TEST(Serving, HttpDisconnectCancelsOnlyActiveRequest) {
  FakeBackend backend;
  backend.block = true;
  Service service(backend, "test");
  HttpServer server(service);
  auto port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(5, 0);
  auto body = request();
  body["stream"] = true;
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/completions";
  req.body = body.dump();
  req.set_header("Content-Type", "application/json");
  req.content_receiver = [](const char *, size_t, uint64_t, uint64_t) {
    return false;
  };
  auto interrupted = client.send(req);
  EXPECT_FALSE(interrupted);
  auto listed = service.submit("GET", "/test-barrier", {});
  auto state = listed.wait_for(std::chrono::seconds(5));
  if (state != std::future_status::ready)
    backend.release.set_value();
  EXPECT_EQ(state, std::future_status::ready);
  listed.wait();
  EXPECT_TRUE(service.submit("GET", "/sessions", {}).get()["data"].empty());
  EXPECT_NO_THROW(call(service, request()));
}
TEST(Serving, BarePostAndControlsStayResponsiveDuringInference) {
  FakeBackend backend;
  Service service(backend, "test", 3, 1);
  auto pinned = service.submit("POST", "/sessions", {}).get()["id"];
  backend.block = true;
  auto entered = backend.entered.get_future();
  auto body = request();
  body["session_id"] = pinned;
  auto running = service.submit("POST", "/v1/completions", body);
  entered.wait();
  // Ensure the inference queue itself is full as well.
  auto waiting = service.submit("POST", "/v1/completions", request());
  struct Release {
    FakeBackend &backend;
    ~Release() { backend.release.set_value(); }
  } release{backend};
  HttpServer server(service);
  auto port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  auto start = std::chrono::steady_clock::now();
  // Exact curl -X POST framing: no body length, no transfer encoding, no EOF.
  auto raw = rawHttp(
    port,
    "POST /sessions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
  EXPECT_NE(raw.find("200 OK"), std::string::npos);
  auto created =
    Json::parse(raw.substr(raw.find("\r\n\r\n") + 4))["id"].get<std::string>();
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(1, 0);
  auto listed = client.Get("/sessions");
  ASSERT_TRUE(listed);
  EXPECT_EQ(listed->status, 200);
  auto entries = Json::parse(listed->body)["data"];
  EXPECT_TRUE(
    std::any_of(entries.begin(), entries.end(), [&](const Json &entry) {
      return entry["id"] == pinned && entry["state"] == "generating";
    }));
  auto busy = client.Delete("/sessions/" + pinned.get<std::string>());
  ASSERT_TRUE(busy);
  EXPECT_EQ(busy->status, 409);
  auto deleted = client.Delete("/sessions/" + created);
  ASSERT_TRUE(deleted);
  EXPECT_EQ(deleted->status, 200);
  auto health = client.Get("/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
  auto emptyCompletion =
    rawHttp(port, "POST /v1/completions HTTP/1.1\r\nHost: "
                  "localhost\r\nConnection: close\r\n\r\n");
  EXPECT_NE(emptyCompletion.find("503 Service Unavailable"), std::string::npos);
}
TEST(Serving, InferenceSaturationReservesHttpWorkersForControls) {
  FakeBackend backend;
  backend.block = true;
  Service service(backend, "test", 32, 32);
  HttpServer server(service);
  auto port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  std::promise<void> rejected;
  auto observed = rejected.get_future();
  std::atomic<bool> signalled{false};
  std::vector<std::future<int>> callers;
  for (int i = 0; i < 7; ++i) {
    callers.push_back(std::async(std::launch::async, [&] {
      httplib::Client client("127.0.0.1", port);
      client.set_read_timeout(10, 0);
      auto response =
        client.Post("/v1/completions", request().dump(), "application/json");
      if (!response)
        return -1;
      if (response->status == 503 && !signalled.exchange(true))
        rejected.set_value();
      return response->status;
    }));
  }
  auto overloaded = observed.wait_for(std::chrono::seconds(3));
  // Avoid blocking test cleanup if the admission invariant regresses.
  if (overloaded != std::future_status::ready)
    backend.release.set_value();
  EXPECT_EQ(overloaded, std::future_status::ready);
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(1, 0);
  auto created = client.Post("/sessions");
  EXPECT_TRUE(created);
  if (created) {
    EXPECT_EQ(created->status, 200);
  }
  auto health = client.Get("/health");
  EXPECT_TRUE(health);
  if (health) {
    EXPECT_EQ(health->status, 200);
  }
  if (overloaded == std::future_status::ready)
    backend.release.set_value();
  for (auto &caller : callers) {
    auto code = caller.get();
    EXPECT_TRUE(code == 200 || code == 503);
  }
}
TEST(Serving, InvalidFramingIsRejectedWithoutWaitingForBody) {
  FakeBackend backend;
  Service service(backend, "test");
  HttpServer server(service);
  auto port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  for (const std::string headers :
       {"Content-Length: 9\r\nTransfer-Encoding: chunked\r\n",
        "Content-Length: 0\r\nContent-Length: 9\r\n",
        "Transfer-Encoding: gzip\r\n"}) {
    auto response =
      rawHttp(port, "POST /sessions HTTP/1.1\r\nHost: localhost\r\n" + headers +
                      "\r\n");
    EXPECT_NE(response.find("400 Bad Request"), std::string::npos);
  }
  auto empty =
    rawHttp(port, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n");
  EXPECT_NE(empty.find("400 Bad Request"), std::string::npos);
  EXPECT_TRUE(service.submit("GET", "/sessions", {}).get()["data"].empty());
}
TEST(Serving, RealPagedCacheInterleavedHistories) {
  setenv("NNTR_ENGINE", "cpu", 1);
  setenv("NNTR_KV_PAGED", "1", 1);
  setenv("NNTR_KV_PREFIX_CACHE", "1", 1);
  const std::string dir = SERVING_FIXTURE_DIR;
  auto load = [&](const std::string &name) {
    Json j;
    std::ifstream file(dir + "/" + name);
    file >> j;
    return j;
  };
  auto cfg = load("config.json"), gen = load("generation_config.json"),
       nntr = load("nntr_config.json");
  nntr["tokenizer_file"] = dir + "/tokenizer.json";
  causallm::Qwen2CausalLM model(cfg, gen, nntr);
  model.initialize();
  model.load_weight(dir + "/nntr_qwen2_tiny_fp32.bin");
  CausalBackend backend(model, nullptr);
  Service service(backend, "test");
  auto body = request();
  body["prompt"] = "a";
  auto a = call(service, body);
  body["prompt"] = "b";
  auto b = call(service, body);
  body["session_id"] = a["session_id"];
  body["prompt"] = "a c";
  auto resumed = call(service, body);
  EXPECT_GT(resumed["usage"]["prompt_tokens"].get<size_t>(),
            a["usage"]["prompt_tokens"].get<size_t>());
  EXPECT_GT(
    resumed["usage"]["prompt_tokens_details"]["cached_tokens"].get<size_t>(),
    0u);
  body["prompt"] = "a";
  auto edited = call(service, body);
  EXPECT_EQ(edited["choices"][0]["text"], a["choices"][0]["text"]);
  service
    .submit("DELETE", "/sessions/" + a["session_id"].get<std::string>(), {})
    .get();
  service
    .submit("DELETE", "/sessions/" + b["session_id"].get<std::string>(), {})
    .get();
  EXPECT_TRUE(service.submit("GET", "/sessions", {}).get()["data"].empty());
  EXPECT_THROW(backend.renderChat({}), ApiError);
}
// Opt-in full-size model test: the model is not a required CI download.
TEST(Serving, Qwen3_06B_NativeHttp) {
  const char *configured = std::getenv("NNTR_SERVING_QWEN3_DIR");
  const std::string dir = configured ? configured : SERVING_QWEN3_DIR;
  if (!std::filesystem::exists(dir + "/nntr_config.json"))
    GTEST_SKIP()
      << "Set NNTR_SERVING_QWEN3_DIR to a converted Qwen3 0.6B directory";
  setenv("NNTR_ENGINE", "cpu", 1);
  setenv("NNTR_KV_PAGED", "1", 1);
  setenv("NNTR_KV_PREFIX_CACHE", "1", 1);
  setenv("NNTR_NUM_THREADS", "8", 1);
  auto load = [&](const std::string &name) {
    Json j;
    std::ifstream file(dir + "/" + name);
    file >> j;
    return j;
  };
  auto cfg = load("config.json"), gen = load("generation_config.json"),
       nntr = load("nntr_config.json");
  nntr["tokenizer_file"] = dir + "/tokenizer.json";
  causallm::Qwen3CausalLM model(cfg, gen, nntr);
  model.initialize();
  model.load_weight(dir + "/" + nntr["model_file_name"].get<std::string>());
  auto chat_template = causallm::ChatTemplate::Load(dir);
  CausalBackend backend(model, &chat_template);
  Service service(backend, "qwen3-0.6b", 8);
  HttpServer server(service);
  int port = server.bind("127.0.0.1", 0);
  ASSERT_GT(port, 0);
  std::thread listener([&] { server.listen(); });
  struct Stop {
    HttpServer &server;
    std::thread &listener;
    ~Stop() {
      server.stop();
      listener.join();
    }
  } stop{server, listener};
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(300, 0);
  Json report = {{"model_dir", dir}, {"engine", "cpu"}, {"storage", "paged"}};
  auto post = [&](const std::string &path, const Json &body,
                  int expected = 200) {
    auto response = client.Post(path, body.dump(), "application/json");
    if (!response)
      throw std::runtime_error("HTTP request failed");
    EXPECT_EQ(response->status, expected) << response->body;
    return Json::parse(response->body);
  };
  auto makeChat = [](const std::string &prompt) {
    return Json{
      {"model", "qwen3-0.6b"},
      {"temperature", 0},
      {"max_tokens", 32},
      {"stream", false},
      {"chat_template_kwargs", {{"enable_thinking", false}}},
      {"messages", Json::array({{{"role", "user"}, {"content", prompt}}})}};
  };
  auto arithmetic = post("/v1/chat/completions", makeChat("Compute 2 + 2."));
  report["arithmetic"] = arithmetic;
  ASSERT_TRUE(arithmetic.contains("choices")) << arithmetic;
  EXPECT_EQ(
    arithmetic["choices"][0]["message"]["content"].get<std::string>().find(
      "<|im_end|>"),
    std::string::npos);
  EXPECT_NE(
    arithmetic["choices"][0]["message"]["content"].get<std::string>().find('4'),
    std::string::npos);
  auto idA = post("/sessions", {})["id"].get<std::string>();
  auto idB = post("/sessions", {})["id"].get<std::string>();
  auto a = makeChat("Remember this word: mango. Reply only OK.");
  a["session_id"] = idA;
  auto b = makeChat("Remember this word: tulip. Reply only OK.");
  b["session_id"] = idB;
  auto firstA = post("/v1/chat/completions", a);
  auto firstB = post("/v1/chat/completions", b);
  report["session_a_first"] = firstA;
  report["session_b_first"] = firstB;
  ASSERT_TRUE(firstA.contains("choices")) << firstA;
  ASSERT_TRUE(firstB.contains("choices")) << firstB;
  for (auto entry :
       {std::make_pair(&a, &firstA), std::make_pair(&b, &firstB)}) {
    (*entry.first)["messages"].push_back(
      (*entry.second)["choices"][0]["message"]);
    (*entry.first)["messages"].push_back(
      {{"role", "user"},
       {"content",
        "What word did I ask you to remember? Reply only with that word."}});
  }
  auto resumedA = post("/v1/chat/completions", a);
  auto resumedB = post("/v1/chat/completions", b);
  report["session_a_resumed"] = resumedA;
  report["session_b_resumed"] = resumedB;
  ASSERT_TRUE(resumedA.contains("choices")) << resumedA;
  ASSERT_TRUE(resumedB.contains("choices")) << resumedB;
  EXPECT_NE(
    resumedA["choices"][0]["message"]["content"].get<std::string>().find(
      "mango"),
    std::string::npos);
  EXPECT_NE(
    resumedB["choices"][0]["message"]["content"].get<std::string>().find(
      "tulip"),
    std::string::npos);
  EXPECT_GT(
    resumedA["usage"]["prompt_tokens_details"]["cached_tokens"].get<size_t>(),
    0u);
  auto fresh = a;
  fresh.erase("session_id");
  auto independent = post("/v1/chat/completions", fresh);
  report["session_a_independent"] = independent;
  ASSERT_TRUE(independent.contains("choices")) << independent;
  EXPECT_EQ(independent["choices"][0]["message"]["content"],
            resumedA["choices"][0]["message"]["content"]);
  auto repeated = post("/v1/chat/completions", a);
  report["session_a_repeated"] = repeated;
  EXPECT_EQ(repeated["choices"][0]["message"]["content"],
            resumedA["choices"][0]["message"]["content"]);
  auto streamedBody = a;
  streamedBody["stream"] = true;
  streamedBody["stream_options"] = {{"include_usage", true}};
  auto streamed = client.Post("/v1/chat/completions", streamedBody.dump(),
                              "application/json");
  ASSERT_TRUE(streamed);
  ASSERT_EQ(streamed->status, 200) << streamed->body;
  EXPECT_EQ(streamed->get_header_value("Content-Type"), "text/event-stream");
  auto chunks = sseEvents(streamed->body);
  ASSERT_GT(chunks.size(), 3u);
  EXPECT_EQ(chunks.back(), "[DONE]");
  std::string joined;
  for (auto &chunk : chunks) {
    if (chunk.is_object() && !chunk["choices"].empty())
      joined += chunk["choices"][0]["delta"].value("content", "");
  }
  EXPECT_EQ(joined, resumedA["choices"][0]["message"]["content"]);
  EXPECT_EQ(joined.find("<|im_end|>"), std::string::npos);
  EXPECT_EQ(chunks[chunks.size() - 2]["usage"]["completion_tokens"],
            resumedA["usage"]["completion_tokens"]);
  report["streaming_chunks"] = chunks;
  auto korean = makeChat("대한민국의 수도는 어디인가요? 한 문장으로 답하세요.");
  auto koreanFull = post("/v1/chat/completions", korean);
  korean["stream"] = true;
  auto koreanStream =
    client.Post("/v1/chat/completions", korean.dump(), "application/json");
  ASSERT_TRUE(koreanStream);
  ASSERT_EQ(koreanStream->status, 200) << koreanStream->body;
  std::string koreanJoined;
  auto koreanChunks = sseEvents(koreanStream->body);
  for (auto &event : koreanChunks)
    if (event.is_object() && !event["choices"].empty())
      koreanJoined += event["choices"][0]["delta"].value("content", "");
  EXPECT_EQ(koreanJoined, koreanFull["choices"][0]["message"]["content"]);
  EXPECT_NE(koreanJoined.find("서울"), std::string::npos);
  report["korean_streaming_text"] = koreanJoined;
  auto cancelId = post("/sessions", {})["id"].get<std::string>();
  auto cancelBody = makeChat("Count from 1 to 100. Write every number.");
  cancelBody["stream"] = true;
  cancelBody["max_tokens"] = 128;
  cancelBody["session_id"] = cancelId;
  httplib::Request cancelReq;
  cancelReq.method = "POST";
  cancelReq.path = "/v1/chat/completions";
  cancelReq.body = cancelBody.dump();
  cancelReq.set_header("Content-Type", "application/json");
  cancelReq.content_receiver = [](const char *, size_t, uint64_t, uint64_t) {
    return false;
  };
  auto disconnected = client.send(cancelReq);
  EXPECT_FALSE(disconnected);
  settle(service);
  auto afterCancel = client.Get("/sessions");
  ASSERT_TRUE(afterCancel);
  auto remaining = Json::parse(afterCancel->body)["data"];
  EXPECT_FALSE(
    std::any_of(remaining.begin(), remaining.end(),
                [&](const Json &entry) { return entry["id"] == cancelId; }));
  auto afterCancelReply = post("/v1/chat/completions", a);
  EXPECT_EQ(afterCancelReply["choices"][0]["message"]["content"],
            resumedA["choices"][0]["message"]["content"]);
  report["cancelled_session_released"] = true;
  auto invalid = a;
  invalid["messages"][0]["content"] = std::string(10000, 'x');
  report["context_rejected"] = post("/v1/chat/completions", invalid, 400);
  auto models = client.Get("/v1/models");
  ASSERT_TRUE(models);
  EXPECT_EQ(models->status, 200);
  report["models"] = Json::parse(models->body);
  auto listed = client.Get("/sessions");
  ASSERT_TRUE(listed);
  auto sessions = Json::parse(listed->body)["data"];
  EXPECT_TRUE(std::any_of(sessions.begin(), sessions.end(),
                          [&](const Json &s) { return s["id"] == idA; }));
  for (auto &session : sessions) {
    auto response =
      client.Delete("/sessions/" + session["id"].get<std::string>());
    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
  }
  auto empty = client.Get("/sessions");
  ASSERT_TRUE(empty);
  report["after_delete"] = Json::parse(empty->body);
  EXPECT_TRUE(report["after_delete"]["data"].empty());
  report["passed"] = !::testing::Test::HasFailure();
  std::ofstream output("/tmp/nntrainer-qwen3-serving-results.json");
  output << report.dump(2);
  std::cout << report.dump(2) << std::endl;
}
} // namespace
