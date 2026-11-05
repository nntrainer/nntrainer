// SPDX-License-Identifier: Apache-2.0
/** @file htp_decode_runtime.h
 * @brief Owns the loaded HTP descriptor and per-session hook/KV progression.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_DECODE_RUNTIME_H
#define NNTR_HTP_DECODE_RUNTIME_H
#include <algorithm>
#include <cstdint>
#include <htp_graph_desc.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace nntrainer {
/** One active model uses the process transport. Weights remain non-owning
 * bindings to the graph; descriptor and session progression live here. */
class HtpDecodeRuntime {
public:
  std::mutex graph_mutex_;
  std::vector<uint32_t> graph_words_;
  std::vector<uint32_t> moe_ops_;
  size_t moe_bound_ = 0;
  std::unordered_map<uint32_t, uint32_t> moe_op_by_handle_;
  /** per MoE op (moe_ops_ order) its EXPERTS table as the
   *  f32 words graph_set_param carries: h_gu[0..E) then h_dn[0..E) */
  std::vector<std::vector<float>> moe_tables_;

  bool graph_inited_ = false;
  bool graph_short_warned_ = false;
  // The stretch tables of section 3.2: per op the maximal resident
  // run it belongs to, per kind the resident ops in list order (what the
  // hooks' counters index), per op whether its parameter is bound and,
  // for CONV1D_GATE, the position its DSP state is valid for; per
  // attention ordinal the DSP cache's length (the kernel's kv_len,
  // mirrored). pending_in_ is the row of the stretch's first op (QK_NORM,
  // ADD, or a MOE op with its routing) until the stretch's last hook runs
  // it (#132's role dispatch). Touched by the model's thread only.
  uint32_t resident_mask_ = 0;
  std::vector<uint32_t> stretch_start_, stretch_end_, attn_ordinal_;
  std::vector<uint32_t> kind_ops_[HTP_OP_KIND_N];
  uint32_t kind_next_[HTP_OP_KIND_N] = {0};
  uint32_t cur_pos_ = HTP_GRAPH_NO_OP;
  bool row_bound_ = false; /**< every MoE op bound when this row began */
  std::vector<uint8_t> param_bound_;
  /** the distinct RoPE tables bound: (ROPE op, its bytes) */
  std::vector<std::pair<uint32_t, std::vector<float>>> rope_tables_;
  /** the host address each bound RoPE table came from, this graph: the
   *  layers of one RoPE cache hand one table at one address over the first
   *  row, and that address names it without a memcmp */
  struct RopePtr {
    const float *data;
    unsigned n;
    uint32_t op;
  };
  std::vector<RopePtr> rope_ptrs_;
  std::vector<uint32_t> conv_next_pos_;
  std::vector<uint32_t> kv_len_;
  bool attn_registered_ = false;
  uint32_t attn_window_rows_ = 0; /**< set_decode_kv_window, 0: none */
  /** per attention ordinal its op, its session cache (0, or
   *  1 for the second attn_m1_register: another (n_kv, gqa, head_dim)) and
   *  its ordinal within that cache (hexkl_graph_init's) */
  std::vector<uint32_t> attn_op_, attn_cache_, attn_cache_ord_;
  /** the f32 parameters the model handed at load
   *  (set_decode_graph_param), bound after graph_init */
  struct LoadParam {
    uint32_t op, which;
    const float *data;
    uint32_t n;
  };
  std::vector<LoadParam> load_params_;
  uint32_t seed_ordinal_ = HTP_GRAPH_NO_OP;
  uint32_t pending_op_ = HTP_GRAPH_NO_OP;
  std::vector<float> pending_in_;
  // the routing a MOE op that starts a stretch was handed
  std::vector<unsigned int> pending_ri_, pending_rc_;
  std::vector<float> pending_rw_;
  // the model's Q4_0 weights for the Q4M1 kinds, in list
  // order, until bindQ4m1; the handles registered for them
  struct Q4Pending {
    const void *data;
    uint32_t K, N;
    bool canonical;
    const float *scale = nullptr; /**< QS4CX: N scales; Q4_0: null */
    bool dropped = false;         /**< CPU copy handed back */
    bool keep_host = false;       /**< CPU prefill still reads this weight */
  };
  std::vector<Q4Pending> q4_pending_;

  std::vector<uint32_t> q4m1_handles_;

  uint32_t first_resident_op_ = HTP_GRAPH_NO_OP;

  uint64_t fwd_calls_ = 0; /**< nntr_hvx_forward* calls */

  uint64_t fwd_tokens_ = 0; /**< of them at the first resident op */

  uint64_t cpu_fc_skipped_ = 0; /**< decode_row_resident's
                                     true answers: CPU FC calls skipped */

  bool want_logits_ = true;

  bool have_id_ = false; /**< take_decode_token_id */

  uint64_t prompt_tokens_ = 0; /**< staging lifetime per prompt */

  uint32_t pending_id_ = 0;

  bool q4m1_bound_ = false; /**< the Q4M1 weights registered at load (E3) */

  size_t q4m1_left_ = 0; /**< E2E FC arena bytes still to place */

  std::vector<uint32_t> ban_, ban_sent_; /**< set_decode_ban, the LM_BAN */
  /** A completed resident row owns the authoritative KV and convolution state.
   *  Initial prefill, an unfinished row and non-contiguous positions keep the
   *  existing graph-walk contract. */
  bool canContinue(unsigned from) const {
    return graph_inited_ && q4m1_bound_ && prompt_tokens_ != 0 &&
           (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) != 0 &&
           cur_pos_ != HTP_GRAPH_NO_OP && cur_pos_ + 1u == from &&
           pending_op_ == HTP_GRAPH_NO_OP && !have_id_;
  }

  void checkContinuationRange(unsigned from, unsigned to) const {
    if (!canContinue(from) || to <= from || to > graph_words_[6])
      throw std::invalid_argument("HTP continuation: invalid token range");
    for (uint32_t op : kind_ops_[HTP_OP_ATTN_M1]) {
      const auto *rec = graphOp(op);
      const uint32_t capacity =
        rec->top_k && attn_window_rows_ ? attn_window_rows_ : graph_words_[6];
      if (from < rec->n_experts || to - rec->n_experts > capacity)
        throw std::invalid_argument(
          "HTP continuation: token range exceeds the reserved KV capacity");
      if (kv_len_[attn_ordinal_[op]] != from - rec->n_experts)
        throw std::logic_error(
          "HTP continuation: attention state is not current");
    }
    for (uint32_t op : kind_ops_[HTP_OP_CONV1D_GATE])
      if (conv_next_pos_[op] != from)
        throw std::logic_error(
          "HTP continuation: convolution state is not current");
  }

  /** Restore the caller's sampling/logits policy, including exceptional exits.
   */
  class LogitsRequest {
  public:
    explicit LogitsRequest(HtpDecodeRuntime &runtime) :
      runtime_(runtime), previous_(runtime.want_logits_) {}
    ~LogitsRequest() { runtime_.want_logits_ = previous_; }
    LogitsRequest(const LogitsRequest &) = delete;
    LogitsRequest &operator=(const LogitsRequest &) = delete;
    void set(bool want) { runtime_.want_logits_ = want; }

  private:
    HtpDecodeRuntime &runtime_;
    bool previous_;
  };

  const htp_graph_op *graphOp(uint32_t op) const {
    return htp_graph_op_cat(graph_words_.data(), op);
  }
  void clearPending() {
    pending_op_ = HTP_GRAPH_NO_OP;
    pending_in_.clear();
    pending_ri_.clear();
    pending_rc_.clear();
    pending_rw_.clear();
  }
  void indexGraph(uint32_t mask) {
    const auto &words = graph_words_;
    const uint32_t n_ops = words[3];
    resident_mask_ = mask;
    moe_ops_.clear();
    for (auto &v : kind_ops_)
      v.clear();
    stretch_start_.assign(n_ops, 0);
    stretch_end_.assign(n_ops, 0);
    attn_ordinal_.assign(n_ops, 0);
    param_bound_.assign(n_ops, 0);
    rope_ptrs_.clear();
    conv_next_pos_.assign(n_ops, HTP_GRAPH_NO_OP);
    first_resident_op_ = HTP_GRAPH_NO_OP;
    uint32_t n_attn = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (op->kind == HTP_OP_MOE)
        moe_ops_.push_back(i);
      if (op->kind == HTP_OP_ATTN_M1)
        attn_ordinal_[i] = n_attn++;
      if (!op->resident)
        continue;
      kind_ops_[op->kind].push_back(i);
      if (first_resident_op_ == HTP_GRAPH_NO_OP)
        first_resident_op_ = i;
      // a stretch is a maximal run of resident ops [s, e)
      stretch_start_[i] =
        (i > 0 && htp_graph_op_cat(words.data(), i - 1)->resident)
          ? stretch_start_[i - 1]
          : i;
    }
    for (uint32_t i = n_ops; i-- > 0;) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (!op->resident)
        continue;
      stretch_end_[i] =
        (i + 1 < n_ops && htp_graph_op_cat(words.data(), i + 1)->resident)
          ? stretch_end_[i + 1]
          : i + 1;
    }
    kv_len_.assign(n_attn, 0);
    // the attention shapes, in list order (one cache each,
    // at most two: Gemma 4's sliding and full layers)
    attn_op_.assign(n_attn, 0);
    attn_cache_.assign(n_attn, 0);
    attn_cache_ord_.assign(n_attn, 0);
    {
      std::vector<const htp_graph_op *> shapes;
      std::vector<uint32_t> per_shape;
      for (uint32_t i = 0; i < n_ops; ++i) {
        const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
        if (op->kind != HTP_OP_ATTN_M1)
          continue;
        size_t c = 0;
        while (c < shapes.size() &&
               (shapes[c]->n_kv != op->n_kv || shapes[c]->gqa != op->gqa ||
                shapes[c]->head_dim != op->head_dim))
          ++c;
        if (c == shapes.size()) {
          if (c == 2u)
            throw std::invalid_argument(
              "set_decode_graph_desc: a third attention shape (the session "
              "holds two caches)");
          shapes.push_back(op);
          per_shape.push_back(0u);
        }
        const uint32_t ord = attn_ordinal_[i];
        attn_op_[ord] = i;
        attn_cache_[ord] = static_cast<uint32_t>(c);
        attn_cache_ord_[ord] = per_shape[c]++;
      }
    }
    load_params_.clear();
    std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
    cur_pos_ = HTP_GRAPH_NO_OP;
    clearPending();
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    attn_registered_ = false;
    moe_bound_ = 0;
    moe_op_by_handle_.clear();
    moe_tables_.assign(moe_ops_.size(), {});
    graph_inited_ = false;
  }

  bool beginRow(unsigned kind, unsigned pos) {
    if (graph_words_.empty() || kind >= HTP_OP_KIND_N ||
        (resident_mask_ & HTP_GRAPH_KIND_BIT(kind)) == 0u)
      return false;
    if (pos != cur_pos_) {
      if (pending_op_ != HTP_GRAPH_NO_OP) {
        throw std::runtime_error(
          "decode_op_fp32: pos moved to " + std::to_string(pos) +
          " with the stretch input of op " + std::to_string(pending_op_) +
          " pending (its last hook never came)");
      }
      std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
      cur_pos_ = pos;
      // Decided once per row, so a row that starts before the MoE handles
      // are all bound (a one-token prompt binds them as it goes) runs on
      // the CPU end to end: a hook after the last MoE call would otherwise
      // resolve to the first op of its kind. The subset case
      // (moe_htp_layers) stays 0 for good; gemm_qs4cx_moe_layer_fp32
      // warned once.
      row_bound_ = moe_bound_ == moe_ops_.size();
    }
    if (!row_bound_)
      return false;
    return true;
  }

  uint32_t nextHook(unsigned kind, unsigned pos) {
    const std::vector<uint32_t> &ops = kind_ops_[kind];
    if (kind_next_[kind] >= ops.size()) {
      throw std::runtime_error(std::string("decode_op_fp32: more ") +
                               htp_graph_kind_name(kind) + " hooks at pos " +
                               std::to_string(pos) + " than the list's " +
                               std::to_string(ops.size()) + " ops");
    }
    return ops[kind_next_[kind]++];
  }

  static constexpr uint32_t kArmKinds =
    HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |
    HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) | HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1) | HTP_GRAPH_KIND_BIT(HTP_OP_ADD) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK) | HTP_GRAPH_KINDS_Q4M1;
  static void validate(const std::vector<uint32_t> &words, uint32_t &n_ops,
                       uint32_t &mask, uint32_t &present) {
    n_ops = 0;
    // The same validator the skel runs, so a bad list fails here with the
    // same code and name before any FastRPC call is made.
    const uint32_t rc = htp_graph_validate(
      words.data(), static_cast<uint32_t>(words.size()), kArmKinds, &n_ops);
    if (rc != 0u) {
      throw std::invalid_argument(std::string("set_decode_graph_desc: ") +
                                  htp_graph_err_name(rc));
    }
    mask = present = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      present |= HTP_GRAPH_KIND_BIT(op->kind);
      if (op->resident)
        mask |= HTP_GRAPH_KIND_BIT(op->kind);
    }
    // The ARM's own rule: a resident QK_NORM or ROPE needs ATTN_M1 --
    // no CPU layer consumes their DSP output (the qkv hook writes
    // nothing; ROPE has no layer), so the stretch must end in the
    // attention hook. The DSP enforces only ATTN_M1 -> ROPE.
    if ((mask & (HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) |
                 HTP_GRAPH_KIND_BIT(HTP_OP_ROPE))) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: QK_NORM / ROPE "
                                  "resident without ATTN_M1 (no CPU layer "
                                  "consumes their output)");
    }
    // A resident ROUTER_TOPK needs ADD resident: then the ffn norm
    // before it and the ADD after its MOE are resident too, so the MoE
    // layer's one hook is always mid-stretch and the layer skips its
    // router, top-k and experts. Without ADD the MoE call would start the
    // stretch, and it has no row of its own to start it with.
    if ((mask & HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK)) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ADD)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: ROUTER_TOPK "
                                  "resident without ADD");
    }
    // No CPU layer hooks an FC, the dense FFN or a
    // projection, so a Q4M1 kind is resident only when every kind the
    // list has is: the list is then one stretch per token, op 0's norm
    // hook keeps the row and the lm_head hook runs it (one call per
    // token). Its weights come from the model at load
    // (add_decode_graph_q4_0).
    if ((mask & HTP_GRAPH_KINDS_Q4M1) != 0u && mask != present) {
      throw std::invalid_argument(
        "set_decode_graph_desc: FC / DENSE_FFN / LM_HEAD resident without "
        "every other kind (one session runs them only as the whole token)");
    }
  }
};
/** Borrow the active runtime without creating a backend or opening the DSP. */
HtpDecodeRuntime *getHtpDecodeRuntime() noexcept;
} // namespace nntrainer
#endif
