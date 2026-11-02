// SPDX-License-Identifier: Apache-2.0
#include "http_server.h"
#include "third_party/cpp-httplib/httplib.h"

namespace causallm::serving {
struct HttpServer::Impl {
  httplib::Server http;
  Service &service;
  explicit Impl(Service &service_) : service(service_) {
    http.new_task_queue = [] { return new httplib::ThreadPool(8, 64); };
    http.set_payload_max_length(1024 * 1024);
    http.set_read_timeout(30, 0);
    http.set_write_timeout(30, 0);
    auto route = [this](const httplib::Request &request,
                        httplib::Response &response) {
      try {
        Json body = Json::object();
        if (!request.body.empty())
          body = Json::parse(request.body);
        auto result =
          service.submit(request.method, request.path, std::move(body)).get();
        response.set_content(result.dump(), "application/json");
      } catch (const ApiError &e) {
        response.status = e.status;
        response.set_content(errorBody(e.status, e.what(), e.code).dump(),
                             "application/json");
      } catch (const Json::exception &) {
        response.status = 400;
        response.set_content(
          errorBody(400, "Malformed JSON request", "invalid_json").dump(),
          "application/json");
      } catch (const std::exception &) {
        response.status = 500;
        response.set_content(
          errorBody(500, "Inference service failed", "internal_error").dump(),
          "application/json");
      }
    };
    http.Get("/health", [](const auto &, auto &res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    http.Get("/v1/models", [this](const auto &, auto &res) {
      res.set_content(
        Json({{"object", "list"},
              {"data", Json::array({{{"id", service.modelId()},
                                     {"object", "model"},
                                     {"created", 0},
                                     {"owned_by", "nntrainer"}}})}})
          .dump(),
        "application/json");
    });
    http.Post("/v1/chat/completions", route);
    http.Post("/v1/completions", route);
    http.Get("/sessions", route);
    http.Post("/sessions", route);
    http.Delete("/sessions/[^/]+", route);
    http.set_error_handler([](const auto &, auto &response) {
      if (response.body.empty())
        response.set_content(
          errorBody(response.status, "HTTP request rejected", "http_error")
            .dump(),
          "application/json");
    });
  }
};
HttpServer::HttpServer(Service &service) :
  impl(std::make_unique<Impl>(service)) {}
HttpServer::~HttpServer() { stop(); }
int HttpServer::bind(const std::string &host, int port) {
  if (port == 0)
    return impl->http.bind_to_any_port(host);
  return impl->http.bind_to_port(host, port) ? port : -1;
}
bool HttpServer::listen() { return impl->http.listen_after_bind(); }
void HttpServer::stop() { impl->http.stop(); }
} // namespace causallm::serving
