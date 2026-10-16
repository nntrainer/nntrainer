// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_token_completion.h
 * @brief Completion hint and one-polling-pass-per-token rules.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_TOKEN_COMPLETION_H
#define NNTR_HTP_TOKEN_COMPLETION_H

#include <atomic>
#include <cstdint>

namespace nntrainer {

/** The server owns this state. Its condition variable may wake before ARM
 * disarms a completed token; that must not restart polling for the same token.
 * Token numbers advance for every submission, including failed submissions.
 */
class HtpTokenPollState {
public:
  bool canStart(bool active, uint32_t tok) const {
    return active && (!started_ || tok != last_token_);
  }

  void begin(uint32_t tok) {
    last_token_ = tok;
    started_ = true;
  }

  /** Load is a volatile mailbox read in production and atomic in host tests.
   * The inverse rejects the cleared {0,0} record, including for token zero.
   * The second sequence read rejects a torn record. The hint permits parking
   * only; it never replaces the authoritative DSPQ status and output. */
  template <typename Load> static bool isDone(uint32_t tok, Load load) {
    if (load(0u) != tok)
      return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint32_t inverse = load(1u);
    return inverse == ~tok && load(0u) == tok;
  }

private:
  uint32_t last_token_ = 0;
  bool started_ = false;
};

} // namespace nntrainer
#endif
