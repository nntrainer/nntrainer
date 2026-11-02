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
    if (block) {
      entered.set_value();
      release.get_future().wait();
      block = false;
    }
    if (fail)
      throw ApiError(500, "failed", "inference_failed");
    if (!caches.count(id))
      throw std::logic_error("Wrong cache");
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
  auto bad = client.Post("/v1/chat/completions", "{", "application/json");
  ASSERT_TRUE(bad);
  EXPECT_EQ(bad->status, 400);
  EXPECT_TRUE(Json::parse(bad->body)["error"].contains("type"));
  auto unknown = client.Get("/missing");
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown->status, 404);
  EXPECT_TRUE(Json::parse(unknown->body).contains("error"));
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
  auto invalid = a;
  invalid["stream"] = true;
  report["stream_rejected"] = post("/v1/chat/completions", invalid, 400);
  invalid = a;
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
