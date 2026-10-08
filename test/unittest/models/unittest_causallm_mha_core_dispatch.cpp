// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_causallm_mha_core_dispatch.cpp
 * @date   30 Sep 2026
 * @brief  MHACoreLayer's dispatch of attention to ComputeOps, against a fake
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The layer hands attention to the context's ComputeOps when it offers
 * sdpa_fp16_kvcache (attention over the fp16 cache) or the kv_cache_q_* /
 * sdpa_q_kvcache family (a quantized mirror of the cache the op keeps).
 * This drives the layer in its external-cache mode through a fake
 * ComputeOps that records every call and answers with an exact f32
 * attention, and checks the plumbing the device cannot: the arguments
 * carry the right rows, window, softcap and sinks; a declined call falls
 * back to the CPU path with the same result; the quantized mirror is
 * re-appended from the first row that may have changed after a rewind or
 * a repositioned (loaded) cache.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <context_data.h>
#include <cpu_ops_table.h>
#include <layer_context.h>
#include <mha_core.h>
#include <var_grad.h>
#include <weight.h>

namespace {

using nntrainer::ComputeOps;
using nntrainer::ContextData;
using nntrainer::InitLayerContext;
using nntrainer::RunLayerContext;
using nntrainer::Var_Grad;
using nntrainer::Weight;

/** @brief IEEE binary16 bits -> f32 (normals and zero suffice here). */
float hf_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t u;
  if (exp == 0) {
    u = sign; // subnormals are not produced by the values used here
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

/** @brief f32 -> binary16 bits, round to nearest even. */
uint16_t f32_to_hf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  const uint32_t rem = mant & 0x1FFFu;
  mant >>= 13;
  if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
    if (++mant == 0x400u) {
      mant = 0;
      ++exp;
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               mant);
}

uint32_t lcg(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return s;
}

float uni(uint32_t &s, float amp) {
  return (static_cast<float>(lcg(s) >> 8) / 16777216.0f * 2.0f - 1.0f) * amp;
}

/**
 * @brief MHACoreLayer's attention, exactly: row q at position cache_from+q
 *        sees cache rows [max(0, pos+1-window), pos]; scores / sqrt(hd),
 *        then tanh(s/softcap)*softcap when softcap > 0; the sink logit
 *        joins the denominator.
 */
void ref_attention(const float *q, unsigned q_stride, const uint16_t *k,
                   const uint16_t *v, unsigned kv_stride, unsigned n_q,
                   unsigned cache_from, unsigned cache_to, unsigned n_head_q,
                   unsigned n_head_kv, unsigned hd, unsigned window,
                   float softcap, const float *sinks, float *out,
                   unsigned out_stride) {
  const unsigned G = n_head_q / n_head_kv;
  const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(hd));
  std::vector<double> sc;
  for (unsigned qi = 0; qi < n_q; ++qi) {
    const unsigned pos = cache_from + qi;
    const unsigned hi = std::min(pos + 1, cache_to);
    const unsigned lo = (window && hi > window) ? hi - window : 0;
    for (unsigned h = 0; h < n_head_q; ++h) {
      const unsigned n = h / G;
      const float *qr = q + static_cast<size_t>(qi) * q_stride + h * hd;
      sc.assign(hi - lo, 0.0);
      double mx = -1e300;
      for (unsigned r = lo; r < hi; ++r) {
        const uint16_t *kr = k + static_cast<size_t>(r) * kv_stride + n * hd;
        double a = 0.0;
        for (unsigned d = 0; d < hd; ++d) {
          a += static_cast<double>(qr[d]) * hf_to_f32(kr[d]);
        }
        a *= inv_sqrt;
        if (softcap > 0.0f) {
          a = std::tanh(a / softcap) * softcap;
        }
        sc[r - lo] = a;
        mx = std::max(mx, a);
      }
      double den = 0.0;
      if (sinks) {
        mx = std::max(mx, static_cast<double>(sinks[h]));
        den = std::exp(sinks[h] - mx);
      }
      for (unsigned r = lo; r < hi; ++r) {
        sc[r - lo] = std::exp(sc[r - lo] - mx);
        den += sc[r - lo];
      }
      float *o = out + static_cast<size_t>(qi) * out_stride + h * hd;
      for (unsigned d = 0; d < hd; ++d) {
        double a = 0.0;
        for (unsigned r = lo; r < hi; ++r) {
          a += sc[r - lo] *
               hf_to_f32(v[static_cast<size_t>(r) * kv_stride + n * hd + d]);
        }
        o[d] = static_cast<float>(a / den);
      }
    }
  }
}

/**
 * @brief A ComputeOps that records the attention calls and answers them
 *        with ref_attention. The quantized family keeps a plain fp16 copy
 *        of the appended rows as its "mirror", so a row the layer failed to
 *        re-append shows up as a wrong answer.
 */
class FakeOps : public nntrainer::CpuComputeOps {
public:
  bool fp16_supported = false;
  bool fp16_declines = false;
  bool q_supported = false;
  bool q_declines = false;

  struct Fp16Call {
    unsigned n_q, cache_from, cache_to, window;
    float softcap;
    std::vector<float> sinks;
  };
  struct QCall {
    int handle;
    unsigned append_row0, append_rows, n_q, cache_from, cache_to;
  };
  std::vector<Fp16Call> fp16_calls;
  std::vector<QCall> q_calls;
  unsigned registers = 0, releases = 0;

  bool supports_sdpa_fp16_kvcache() const override { return fp16_supported; }

  bool sdpa_fp16_kvcache(const float *q, unsigned q_stride,
                         const uint16_t *k_cache, const uint16_t *v_cache,
                         unsigned kv_stride, unsigned n_q, unsigned cache_from,
                         unsigned cache_to, unsigned n_head_q,
                         unsigned n_head_kv, unsigned head_dim, unsigned window,
                         float softcap, const float *sinks, float *out,
                         unsigned out_stride) override {
    Fp16Call c{n_q, cache_from, cache_to, window, softcap, {}};
    if (sinks) {
      c.sinks.assign(sinks, sinks + n_head_q);
    }
    fp16_calls.push_back(c);
    if (fp16_declines) {
      return false;
    }
    ref_attention(q, q_stride, k_cache, v_cache, kv_stride, n_q, cache_from,
                  cache_to, n_head_q, n_head_kv, head_dim, window, softcap,
                  sinks, out, out_stride);
    return true;
  }

  bool supports_kv_cache_q() const override { return q_supported; }

  int kv_cache_q_register(unsigned kind, unsigned max_rows, unsigned n_head_kv,
                          unsigned head_dim) override {
    if (kind > 1) {
      return -1;
    }
    ++registers;
    Mirror m;
    m.stride = n_head_kv * head_dim;
    m.k.assign(static_cast<size_t>(max_rows) * m.stride, 0);
    m.v.assign(m.k.size(), 0);
    mirrors.push_back(m);
    return static_cast<int>(mirrors.size()) - 1;
  }

  bool kv_cache_q_append(int handle, unsigned row0, unsigned n_rows,
                         unsigned kv_stride, const uint16_t *k_rows,
                         const uint16_t *v_rows) override {
    if (handle < 0 || static_cast<size_t>(handle) >= mirrors.size()) {
      return false;
    }
    Mirror &m = mirrors[handle];
    if (kv_stride != m.stride ||
        (static_cast<size_t>(row0) + n_rows) * m.stride > m.k.size()) {
      return false;
    }
    std::memcpy(&m.k[static_cast<size_t>(row0) * m.stride], k_rows,
                static_cast<size_t>(n_rows) * m.stride * 2);
    std::memcpy(&m.v[static_cast<size_t>(row0) * m.stride], v_rows,
                static_cast<size_t>(n_rows) * m.stride * 2);
    return true;
  }

  void kv_cache_q_release(int handle) override {
    (void)handle;
    ++releases;
  }

  bool sdpa_q_kvcache(int handle, unsigned append_row0, unsigned append_rows,
                      unsigned kv_stride, const uint16_t *k_rows,
                      const uint16_t *v_rows, const float *q, unsigned q_stride,
                      unsigned n_q, unsigned cache_from, unsigned cache_to,
                      unsigned n_head_q, unsigned n_head_kv, unsigned head_dim,
                      unsigned window, float softcap, const float *sinks,
                      float *out, unsigned out_stride) override {
    q_calls.push_back(
      {handle, append_row0, append_rows, n_q, cache_from, cache_to});
    if (q_declines) {
      return false;
    }
    if (append_rows && !kv_cache_q_append(handle, append_row0, append_rows,
                                          kv_stride, k_rows, v_rows)) {
      return false;
    }
    const Mirror &m = mirrors[handle];
    ref_attention(q, q_stride, m.k.data(), m.v.data(), m.stride, n_q,
                  cache_from, cache_to, n_head_q, n_head_kv, head_dim, window,
                  softcap, sinks, out, out_stride);
    return true;
  }

private:
  struct Mirror {
    unsigned stride;
    std::vector<uint16_t> k, v;
  };
  std::vector<Mirror> mirrors;
};

struct Params {
  unsigned n_head_q = 4, n_head_kv = 2, head_dim = 32, max_rows = 64;
  unsigned window = 0; /**< 0: none */
  float softcap = 0.0f;
  bool sink = false;
  std::string quant; /**< "", "q8" */
};

/** @brief Rows of Q/K/V the input tensors hold; a step uses the first n. */
constexpr unsigned kMaxStep = 16;

/**
 * @brief One MHACoreLayer in external-cache mode (inputs Q, K, V,
 *        cache_key, cache_value) with its own tensors and a RunLayerContext
 *        whose ContextData carries @a ops (nullptr: the plain CPU path).
 */
struct Rig {
  Params p;
  causallm::MHACoreLayer layer;
  std::vector<Weight> weights;
  std::vector<Var_Grad> ins, outs, tensors;
  std::shared_ptr<ContextData> cd;
  std::unique_ptr<RunLayerContext> rc;

  Rig(const Params &params, ComputeOps *ops) : p(params) {
    std::vector<std::string> props = {
      "num_heads=" + std::to_string(p.n_head_q),
      "num_heads_KV=" + std::to_string(p.n_head_kv),
      "max_timestep=" + std::to_string(p.max_rows), "use_rope=false",
      "is_causal=true"};
    if (p.window) {
      props.push_back("sliding_window=" + std::to_string(p.window));
    }
    if (p.softcap > 0.0f) {
      props.push_back("attn_logit_softcapping=" + std::to_string(p.softcap));
    }
    if (p.sink) {
      props.push_back("use_sink=true");
    }
    if (!p.quant.empty()) {
      props.push_back("kv_cache_quant=" + p.quant);
    }
    layer.setProperty(props);

    using DT = ml::train::TensorDim::DataType;
    const nntrainer::TensorDim::TensorType f32(
      ml::train::TensorDim::Format::NCHW, DT::FP32);
    const nntrainer::TensorDim::TensorType u16(
      ml::train::TensorDim::Format::NCHW, DT::UINT16);
    const nntrainer::TensorDim q_dim({1, 1, kMaxStep, p.n_head_q * p.head_dim},
                                     f32);
    const nntrainer::TensorDim kv_dim(
      {1, 1, kMaxStep, p.n_head_kv * p.head_dim}, f32);
    const nntrainer::TensorDim cache_dim(
      {1, 1, p.max_rows, p.n_head_kv * p.head_dim}, u16);
    InitLayerContext ic({q_dim, kv_dim, kv_dim, cache_dim, cache_dim}, {true},
                        false, "mha", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f,
                        ml::train::ExecutionMode::INFERENCE);
    layer.finalize(ic);

    for (auto &spec : ic.getWeightsSpec()) {
      weights.emplace_back(spec, true);
    }
    for (auto &dim : ic.getInputDimensions()) {
      ins.emplace_back(dim, nntrainer::Initializer::NONE, false, true, "in");
    }
    for (auto &spec : ic.getOutSpecs()) {
      outs.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                        false, true, "out");
    }
    for (auto &spec : ic.getTensorsSpec()) {
      tensors.emplace_back(spec, true);
    }
    for (auto &t : ins) {
      std::memset(t.getVariableRef().getData<char>(), 0,
                  t.getVariableRef().bytes());
    }
    if (ops) {
      cd = std::make_shared<ContextData>();
      cd->setComputeOps(ops);
    }
    auto view = [](auto &vec) {
      std::vector<std::remove_reference_t<decltype(vec[0])> *> ret;
      for (auto &x : vec) {
        ret.push_back(&x);
      }
      return ret;
    };
    rc = std::make_unique<RunLayerContext>("mha", false, 0.0f, false, 1.0f, cd,
                                           false, view(weights), view(ins),
                                           view(outs), view(tensors));
  }

  nntrainer::Tensor &cache(unsigned which) {
    return ins[3 + which].getVariableRef();
  }

  /** @brief Per-head sink logits, when the layer has them. */
  void set_sinks(const std::vector<float> &s) {
    float *w = weights[0].getVariableRef().getData<float>();
    std::copy(s.begin(), s.end(), w);
  }

  /**
   * @brief Fills Q/K/V rows for positions [from, to) from @a seed and runs
   *        the step; returns the output rows.
   */
  std::vector<float> step(unsigned from, unsigned to, uint32_t seed) {
    const unsigned n = to - from;
    for (unsigned i = 0; i < 3; ++i) {
      nntrainer::Tensor &t = ins[i].getVariableRef();
      float *d = t.getData<float>();
      uint32_t s = seed * 7919u + i * 104729u + from;
      for (size_t k = 0; k < static_cast<size_t>(n) * t.width(); ++k) {
        d[k] = uni(s, 1.0f);
      }
    }
    layer.incremental_forwarding(*rc, from, to, false);
    const nntrainer::Tensor &o = rc->getOutput(0);
    const float *od = o.getData<float>();
    return std::vector<float>(od, od + static_cast<size_t>(n) * o.width());
  }

  /** @brief Overwrites cache rows [0, rows) of both caches with fp16 noise. */
  void rewrite_cache(unsigned rows, uint32_t seed) {
    for (unsigned c = 0; c < 2; ++c) {
      nntrainer::Tensor &t = cache(c);
      uint16_t *d = t.getData<uint16_t>();
      uint32_t s = seed + 31u * c;
      for (size_t k = 0; k < static_cast<size_t>(rows) * t.width(); ++k) {
        d[k] = f32_to_hf(uni(s, 1.0f));
      }
    }
  }
};

void expect_close(const std::vector<float> &a, const std::vector<float> &b,
                  const char *what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  EXPECT_LT(max_diff, 2e-3) << what;
}

/** @brief Prefill 8 rows, then two decode steps, on both rigs; compares. */
void run_and_compare(Rig &cpu, Rig &acc, const char *what) {
  const unsigned steps[][2] = {{0, 8}, {8, 9}, {9, 10}};
  for (const auto &s : steps) {
    SCOPED_TRACE(std::string(what) + " step " + std::to_string(s[0]) + ".." +
                 std::to_string(s[1]));
    const auto a = cpu.step(s[0], s[1], 11u);
    const auto b = acc.step(s[0], s[1], 11u);
    expect_close(a, b, what);
  }
}

} // namespace

TEST(MhaCoreDispatch, Fp16PathMatchesCpu) {
  Params p;
  FakeOps ops;
  ops.fp16_supported = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "plain");
  ASSERT_EQ(ops.fp16_calls.size(), 3u);
  EXPECT_EQ(ops.fp16_calls[0].n_q, 8u);
  EXPECT_EQ(ops.fp16_calls[0].cache_to, 8u);
  EXPECT_EQ(ops.fp16_calls[2].cache_from, 9u);
  EXPECT_EQ(ops.fp16_calls[2].cache_to, 10u);
  EXPECT_EQ(ops.fp16_calls[0].window, 0u);
}

TEST(MhaCoreDispatch, Fp16WindowAndSoftcapMatchCpu) {
  Params p;
  p.window = 4;
  p.softcap = 5.0f;
  FakeOps ops;
  ops.fp16_supported = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "window+softcap");
  ASSERT_FALSE(ops.fp16_calls.empty());
  // The layer spells "no window" 0 and passes a real window through.
  EXPECT_EQ(ops.fp16_calls.back().window, 4u);
  EXPECT_FLOAT_EQ(ops.fp16_calls.back().softcap, 5.0f);
}

TEST(MhaCoreDispatch, SinkWeightArrivesAsF32) {
  Params p;
  p.sink = true;
  FakeOps ops;
  ops.fp16_supported = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  // Far below any score, so a CPU path that ignores the sink (this host's
  // fallback softmax does) agrees with the fake, which honours it; the
  // values themselves are checked on the call.
  const std::vector<float> sinks = {-40.0f, -41.0f, -42.0f, -43.0f};
  cpu.set_sinks(sinks);
  acc.set_sinks(sinks);
  run_and_compare(cpu, acc, "sink");
  ASSERT_FALSE(ops.fp16_calls.empty());
  ASSERT_EQ(ops.fp16_calls.back().sinks.size(), 4u);
  for (unsigned h = 0; h < 4; ++h) {
    EXPECT_FLOAT_EQ(ops.fp16_calls.back().sinks[h], sinks[h]);
  }
}

TEST(MhaCoreDispatch, DeclinedFp16CallFallsBackToCpu) {
  Params p;
  FakeOps ops;
  ops.fp16_supported = true;
  ops.fp16_declines = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "declined");
  // Asked every step, refused every step, output from the CPU path.
  EXPECT_EQ(ops.fp16_calls.size(), 3u);
}

TEST(MhaCoreDispatch, QuantizedMirrorAppendsOnlyNewRows) {
  Params p;
  p.quant = "q8";
  FakeOps ops;
  ops.q_supported = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "q8");
  ASSERT_EQ(ops.registers, 1u);
  ASSERT_EQ(ops.q_calls.size(), 3u);
  EXPECT_EQ(ops.q_calls[0].append_row0, 0u);
  EXPECT_EQ(ops.q_calls[0].append_rows, 8u);
  EXPECT_EQ(ops.q_calls[1].append_row0, 8u);
  EXPECT_EQ(ops.q_calls[1].append_rows, 1u);
  EXPECT_EQ(ops.q_calls[2].append_row0, 9u);
  EXPECT_EQ(ops.q_calls[2].append_rows, 1u);
}

TEST(MhaCoreDispatch, QuantizedMirrorFollowsRewindAndRepositionedCache) {
  Params p;
  p.quant = "q8";
  FakeOps ops;
  ops.q_supported = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "q8 before");

  // Rewind: the host steps again from an earlier position. Rows from there
  // on are rewritten, so the mirror must be re-appended from there.
  {
    const auto a = cpu.step(4, 5, 21u);
    const auto b = acc.step(4, 5, 21u);
    expect_close(a, b, "rewind");
    ASSERT_FALSE(ops.q_calls.empty());
    EXPECT_EQ(ops.q_calls.back().append_row0, 4u);
    EXPECT_EQ(ops.q_calls.back().append_rows, 1u);
  }

  // A repositioned cache (KVCacheManager after load_kvcache): every row
  // below the new position may differ from what the mirror holds, and the
  // layer only learns of it through setCacheIndex. Both rigs get the same
  // new rows [0, 6) and the same position.
  cpu.rewrite_cache(6, 77u);
  acc.rewrite_cache(6, 77u);
  cpu.layer.setCacheIndex(6);
  acc.layer.setCacheIndex(6);
  {
    const auto a = cpu.step(6, 7, 31u);
    const auto b = acc.step(6, 7, 31u);
    expect_close(a, b, "after setCacheIndex");
    EXPECT_EQ(ops.q_calls.back().append_row0, 0u);
    EXPECT_EQ(ops.q_calls.back().append_rows, 7u);
  }
  // And the step after that appends just its own row again.
  {
    const auto a = cpu.step(7, 8, 41u);
    const auto b = acc.step(7, 8, 41u);
    expect_close(a, b, "after reposition, next step");
    EXPECT_EQ(ops.q_calls.back().append_row0, 7u);
    EXPECT_EQ(ops.q_calls.back().append_rows, 1u);
  }
}

TEST(MhaCoreDispatch, DeclinedQuantizedCallReleasesAndStaysOnCpu) {
  Params p;
  p.quant = "q8";
  FakeOps ops;
  ops.q_supported = true;
  ops.q_declines = true;
  Rig cpu(p, nullptr), acc(p, &ops);
  run_and_compare(cpu, acc, "q8 declined");
  // One attempt, then the handle is released and the layer never asks again.
  EXPECT_EQ(ops.q_calls.size(), 1u);
  EXPECT_EQ(ops.releases, 1u);
}
