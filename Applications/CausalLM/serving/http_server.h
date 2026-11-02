// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "service.h"
#include <memory>

namespace causallm::serving {
/** Native HTTP transport. Bind before running, stop/join before destroying.
 * The bounded HTTP pool only parses requests and waits for service results.
 */
class HttpServer {
public:
  explicit HttpServer(Service &service);
  ~HttpServer();
  int bind(const std::string &host, int port);
  bool listen();
  void stop();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace causallm::serving
