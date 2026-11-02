// SPDX-License-Identifier: Apache-2.0
/** @file graph_binding_check.cpp
 * @brief Verify semantic binding roles against the existing model weight order.
 * @author NNTrainer contributors
 * @bug DSP arithmetic is covered by device tests.
 */
#include <cassert>
#include <cstdio>
#include <htp_graph_binding.h>

using Role = nntrainer::HtpBindingRole;

static void check(const std::vector<uint32_t> &words,
                  const std::vector<Role> &expected) {
  const auto before = words;
  std::vector<Role> actual;
  for (unsigned i = 0; i < words[3]; ++i) {
    auto role = nntrainer::htpBindingRole(words, i);
    if (role != Role::None)
      actual.push_back(role);
  }
  assert(actual == expected);
  assert(before == words); // Binding metadata never changes the descriptor.
  bool rejected = false;
  try {
    nntrainer::htpBindingRole(words, words[3]);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  for (unsigned experts : {32u, 64u}) {
    htp_graph_lfm2_shape lfm{};
    lfm.n_layers = 3;
    lfm.n_dense_layers = 1;
    lfm.hidden = lfm.inter_moe = lfm.vocab = 128;
    lfm.inter_dense = 256;
    lfm.n_experts = experts;
    lfm.top_k = experts / 16;
    lfm.n_heads = 2;
    lfm.n_kv_heads = 1;
    lfm.head_dim = 64;
    lfm.max_seq = 128;
    lfm.eps = 1e-5f;
    const uint8_t attention[] = {0, 1, 0};
    std::vector<uint32_t> words(htp_graph_words_for(3, HTP_GRAPH_MAX_OPS));
    auto n = htp_graph_lfm2_build(words.data(), words.size(), &lfm, attention,
                                  HTP_GRAPH_KINDS_ALL);
    assert(n);
    words.resize(n);
    check(words, {Role::InputNorm, Role::ConvInput, Role::ConvOutput,
                  Role::DenseInputNorm, Role::InputNorm, Role::AttentionInput,
                  Role::AttentionOutput, Role::MoeInputNorm, Role::InputNorm,
                  Role::ConvInput, Role::ConvOutput, Role::MoeInputNorm,
                  Role::OutputNorm});
    for (unsigned k_eq_v : {0u, 1u}) {
      htp_graph_gemma_shape gemma{};
      gemma.n_layers = 2;
      gemma.hidden = gemma.inter_moe = gemma.vocab = 128;
      gemma.inter_dense = 256;
      gemma.n_experts = experts;
      gemma.top_k = experts / 16;
      gemma.n_heads = 4;
      gemma.n_kv = gemma.n_kv_full = 1;
      gemma.head_dim = 64;
      gemma.head_dim_full = 128;
      gemma.k_eq_v = k_eq_v;
      gemma.window = 64;
      gemma.max_seq = 128;
      gemma.eps = 1e-5f;
      const uint8_t full[] = {0, 1};
      words.resize(htp_graph_words_for(2, HTP_GRAPH_MAX_OPS));
      n = htp_graph_gemma_build(words.data(), words.size(), &gemma, full,
                                nullptr, HTP_GRAPH_KINDS_ALL);
      assert(n);
      words.resize(n);
      std::vector<Role> expected;
      // The previous loader's seven norm roles and two FC roles per layer.
      for (unsigned layer = 0; layer < 2; ++layer) {
        const std::vector<Role> roles = {
          Role::InputNorm,       Role::AttentionInput,
          Role::AttentionOutput, Role::AttentionOutputNorm,
          Role::MoeInputNorm,    Role::MoeOutputNorm,
          Role::DenseInputNorm,  Role::DenseOutputNorm,
          Role::FfnOutputNorm};
        expected.insert(expected.end(), roles.begin(), roles.end());
      }
      expected.push_back(Role::OutputNorm);
      check(words, expected);
    }
  }
  std::puts("HTP Gemma/LFM typed bindings preserve weight order PASS");
}
