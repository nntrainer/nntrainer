// SPDX-License-Identifier: Apache-2.0
/** @file decode_runtime_check.cpp
 * @brief Descriptor indexing and hook/session progression without an SDK.
 * @author NNTrainer contributors
 * @bug DSP arithmetic is covered by device tests.
 */
#include <cassert>
#include <cstdio>
#include <htp_decode_runtime.h>

int main() {
  assert(nntrainer::HtpDecodeRuntime::reserveRows(1) == 64);
  assert(nntrainer::HtpDecodeRuntime::reserveRows(64) == 64);
  assert(nntrainer::HtpDecodeRuntime::reserveRows(65) == 128);
  for (uint64_t bad : {uint64_t(0), UINT64_MAX}) {
    bool rejected = false;
    try {
      (void)nntrainer::HtpDecodeRuntime::reserveRows(bad);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
  }
  for (unsigned experts : {32u, 64u}) {
    htp_graph_lfm2_shape shape{};
    shape.n_layers = 3;
    shape.n_dense_layers = 1;
    shape.hidden = shape.inter_moe = shape.vocab = 128;
    shape.inter_dense = 256;
    shape.n_experts = experts;
    shape.top_k = experts / 16;
    shape.n_heads = 2;
    shape.n_kv_heads = 1;
    shape.head_dim = 64;
    shape.max_seq = 128;
    shape.eps = 1e-5f;
    uint8_t attention[] = {0, 1, 0};
    nntrainer::HtpDecodeRuntime runtime;
    auto &words = runtime.graph_words_;
    words.resize(htp_graph_words_for(shape.n_layers, HTP_GRAPH_MAX_OPS));
    const auto n = htp_graph_lfm2_build(words.data(), words.size(), &shape,
                                        attention, HTP_GRAPH_KINDS_ALL);
    assert(n);
    words.resize(n);
    uint32_t ops, mask, present;
    runtime.validate(words, ops, mask, present);
    assert(mask == present);
    runtime.indexGraph(mask);
    assert(runtime.moe_ops_.size() == 2 && runtime.kv_len_.size() == 1);
    assert(runtime.kind_ops_[HTP_OP_CONV1D_GATE].size() == 2);
    // A partially bound first row stays on the original graph walk.
    assert(!runtime.beginRow(HTP_OP_RMSNORM, 0));
    runtime.moe_bound_ = 2;
    assert(!runtime.beginRow(HTP_OP_RMSNORM, 0));
    assert(runtime.beginRow(HTP_OP_RMSNORM, 1));
    const auto first = runtime.nextHook(HTP_OP_RMSNORM, 1);
    assert(first == 0);
    runtime.pending_op_ = first;
    bool threw = false;
    try {
      runtime.beginRow(HTP_OP_RMSNORM, 2);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw); // An unfinished stretch cannot advance the KV position.
    runtime.clearPending();
    runtime.kv_len_[0] = 2;
    runtime.conv_next_pos_[runtime.kind_ops_[HTP_OP_CONV1D_GATE][0]] = 2;
    assert(runtime.beginRow(HTP_OP_RMSNORM, 2));
    assert(runtime.nextHook(HTP_OP_RMSNORM, 2) == first);
    assert(runtime.kv_len_[0] == 2); // Hook counters never reset session KV.
    assert(runtime.conv_next_pos_[runtime.kind_ops_[HTP_OP_CONV1D_GATE][0]] ==
           2);
    assert(!runtime.canContinue(3));
    runtime.graph_inited_ = runtime.q4m1_bound_ = true;
    runtime.prompt_tokens_ = 1;
    runtime.kv_len_[0] = 3;
    for (auto op : runtime.kind_ops_[HTP_OP_CONV1D_GATE])
      runtime.conv_next_pos_[op] = 3;
    assert(runtime.canContinue(3));
    runtime.checkContinuationRange(3, 6);
    runtime.attn_capacity_[0] = 3;
    bool capacity_rejected = false;
    try {
      runtime.checkContinuationRange(3, 6);
    } catch (const std::invalid_argument &) {
      capacity_rejected = true;
    }
    assert(capacity_rejected);
    runtime.checkContinuationRange(3, 6, false);
    runtime.attn_capacity_[0] = 0;
    assert(!runtime.canContinue(2) && !runtime.canContinue(4));
    runtime.have_id_ = true;
    assert(!runtime.canContinue(3));
    runtime.have_id_ = false;
    runtime.pending_op_ = 0;
    assert(!runtime.canContinue(3));
    runtime.clearPending();
    runtime.want_logits_ = false;
    try {
      nntrainer::HtpDecodeRuntime::LogitsRequest request(runtime);
      request.set(true);
      runtime.checkContinuationRange(3, shape.max_seq + 1);
      assert(false);
    } catch (const std::invalid_argument &) {
      assert(!runtime.want_logits_);
    }
    const auto conv_op = runtime.kind_ops_[HTP_OP_CONV1D_GATE][0];
    runtime.conv_next_pos_[conv_op] = 0;
    threw = false;
    try {
      runtime.checkContinuationRange(3, 6);
    } catch (const std::logic_error &) {
      threw = true;
    }
    assert(threw); // Never reseed a stale host convolution state.
    runtime.conv_next_pos_[conv_op] = 3;
    runtime.kv_len_[0] = 0;
    threw = false;
    try {
      runtime.checkContinuationRange(3, 6);
    } catch (const std::logic_error &) {
      threw = true;
    }
    assert(threw); // Never reseed a stale or released host KV cache.
    runtime.indexGraph(mask);
    assert(!runtime.canContinue(3));
    assert(runtime.kv_len_[0] == 0);
    assert(runtime.cur_pos_ == HTP_GRAPH_NO_OP);
    assert(!runtime.beginRow(HTP_OP_RMSNORM, 0));
  }
  std::puts("HTP decode descriptor / binding latch / hook progression PASS");
}
