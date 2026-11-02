// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "causal_lm.h"
#include "chat_template.h"
#include "service.h"

namespace causallm::serving {
/** Adapter to the existing paged CPU KV manager. Model and template outlive
 * the service. The default cache session remains empty for safe destruction.
 */
class CausalBackend final : public Backend {
  CausalLM &model;
  const ChatTemplate *chat_template;

public:
  CausalBackend(CausalLM &model_, const ChatTemplate *template_);
  std::string renderChat(const Json &request) override;
  uint64_t create(const std::string &prompt) override;
  void destroy(uint64_t cache) override;
  Generated generate(uint64_t cache, const Generation &request) override;
};
} // namespace causallm::serving
