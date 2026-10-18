#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay production LFM2 early-prefill scheduling against a captured route.

The actual helper, collection/queue code, model hook and ExpertLru are compiled
with a fake I/O backend. This validates ownership and cache/registration order;
it does not validate DSP arithmetic. Route paths are explicit CLI inputs.

Example (from the repository root):
  python3 test/htp/host/early_prefill_check.py --model lfm2 \
    --routes /path/to/lfm2-prefill.routes
Add --sanitize for ASan/UBSan and --decode-routes for a captured future stream.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=('lfm2',), default='lfm2')
parser.add_argument('--routes', type=Path, required=True)
parser.add_argument('--decode-routes', type=Path)
parser.add_argument('--sanitize', action='store_true')
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
for path in (args.routes, args.decode_routes):
    if path is not None and not path.is_file():
        parser.error(f'Routing trace is not a file: {path}')
root = Path(__file__).resolve().parents[3]
app = root / 'Applications/CausalLM/models'
source_text = (app / 'lfm2_moe/lfm2_moe_layer.cpp').read_text()
model_name = 'Lfm2HtpMoECausalLM'
model_file = 'lfm2_moe/lfm2_htp_moe_causallm.cpp'
model_text = (app / model_file).read_text()


def function(text, start):
    """Extract one balanced production function/block, including its name."""
    begin = text.index(start)
    brace = text.index('{', begin)
    level = 1
    end = brace + 1
    while level:
        level += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[begin:end]


helper_start = source_text.index('static causallm::ExpertLru g_expert_lru;')
helper_end = source_text.index('/**\n * @brief NNTR_MOE_TRACE')
helper = source_text[helper_start:helper_end]
first_start = source_text.index('    auto take_batch = [&] {')
first_end = source_text.index('\n    stage(active);', first_start)
collect = source_text[first_start:first_end]
queue_start = first_end + len('\n    stage(active);')
queue_end = source_text.index('\n    call(output.getData<float>()', queue_start)
queue = source_text[queue_start:queue_end]
hook = function(model_text, 'void ' + model_name + '::beforePrefillBlock(')

layers, experts, hidden, intermediate, early_batches = 38, 64, 2048, 1536, 7
config = (f'constexpr int LAYERS={layers}, EXPERTS={experts}, HIDDEN={hidden}, '
          f'INTERMEDIATE={intermediate}, CAPACITY=LAYERS*16, '
          f'EARLY_BATCHES={early_batches}, '
          'EARLY_READS=EARLY_BATCHES*EXPERTS;\n')

header = r'''

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <expert_lru.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#define HTP_TL2_WEIGHT_BITS 0x102u
#define ENABLE_HEXKL 1
namespace nntrainer {
class ComputeOps {
public:
  struct ExpertFileDesc {
    const void *key_gu;
    unsigned w_bits, K, inter, N_out;
  };
  virtual ~ComputeOps() = default;
  virtual bool
  prefetch_qs4cx_wh_experts_begin(const std::vector<ExpertFileDesc> &) = 0;
  virtual std::vector<const void *> prefetch_qs4cx_wh_experts_end() = 0;
};
ComputeOps *current_ops;
ComputeOps *get_htp_ops() { return current_ops; }
}
namespace causallm {
'''

fixture = r'''

std::array<int, LAYERS * EXPERTS> identities;
int id(const void *p) {
  return static_cast<const int *>(p) - identities.data();
}
std::vector<int> ids(const std::vector<const void *> &p) {
  std::vector<int> result;
  for (auto k : p)
    result.push_back(id(k));
  return result;
}
struct Fake : nntrainer::ComputeOps {
  int layer = -1, demand_reads = 0, end_calls = 0, fail_begin = -1;
  bool refuse = false, workers_stopped = false, old_pass = false;
  std::deque<std::vector<const void *>> batches;
  std::vector<std::vector<int>> reads, registrations;
  std::vector<int> evictions;
  std::set<const void *> resident;
  bool prefetch_qs4cx_wh_experts_begin(
    const std::vector<ExpertFileDesc> &ds) override {
    if (int(reads.size()) == fail_begin) {
      if (refuse)
        return false;
      throw std::runtime_error("injected queue failure");
    }
    std::vector<const void *> keys;
    for (const auto &d : ds) {
      assert(!resident.count(d.key_gu));
      keys.push_back(d.key_gu);
    }
    assert(!keys.empty());
    reads.push_back(ids(keys));
    batches.push_back(keys);
    return true;
  }
  std::vector<const void *> prefetch_qs4cx_wh_experts_end() override {
    assert(!batches.empty());
    auto keys = batches.front();
    batches.pop_front();
    // Only the established old-pass drain may join a future layer early.
    assert(old_pass || id(keys.front()) / EXPERTS <= layer);
    ++end_calls;
    registrations.push_back(ids(keys));
    for (auto k : keys)
      resident.insert(k);
    return keys;
  }
  void evict(const void *k) {
    assert(resident.erase(k) == 1);
    evictions.push_back(id(k));
  }
  void stopWorkers() {
    workers_stopped = true;
    batches.clear();
  }
};

void seed(Fake &ops, size_t preload = 128, size_t capacity_per_layer = 16) {
  resetLfm2HtpExpertPool();
  for (int l = 0; l < LAYERS; ++l) {
    g_expert_lru.addLayer(&identities[l * EXPERTS], capacity_per_layer);
    std::vector<ExpertFileDesc> ds;
    for (int e = 0; e < EXPERTS; ++e)
      ds.push_back({&identities[l * EXPERTS + e], HTP_TL2_WEIGHT_BITS, HIDDEN,
                    INTERMEDIATE, HIDDEN});
    g_expert_layers.push_back(ds);
  }
  for (int l = 0; l < LAYERS && preload; ++l) {
    std::vector<const void *> keys;
    for (int e = 0; e < EXPERTS && preload; ++e, --preload)
      keys.push_back(&identities[l * EXPERTS + e]);
    g_expert_lru.acquire(
      keys, [&](const void *k) { ops.resident.insert(k); },
      [](const void *) { assert(false); });
  }
  g_expert_owner = &ops;
  nntrainer::current_ops = &ops;
}

void forwardLayer(Fake &backend, int expert_layer_slot,
                  const std::vector<int> &active,
                  const std::vector<int> &recency) {
  nntrainer::ComputeOps *ops = &backend;
  backend.layer = expert_layer_slot;
  const int depth = expertPrefetchDepth();
  std::vector<const void *> need;
  for (int e : active)
    need.push_back(&identities[expert_layer_slot * EXPERTS + e]);
  auto release = [&](const void *k) { backend.evict(k); };
'''

stage = r'''

// Ordinary stage(active): exact ExpertLru acquire semantics, fake IO only.
g_expert_lru.acquire(
  need,
  [&](const void *k) {
    ++backend.demand_reads;
    backend.resident.insert(k);
  },
  release);
'''

tail = r'''

std::vector<const void *> refresh;
for (int e : recency)
  refresh.push_back(&identities[expert_layer_slot * EXPERTS + e]);
g_expert_lru.refresh(refresh);
}
'''

model_stub = r'''

class MODEL_NAME {
public:
  bool prepared_ = true, prefill_early_attempted_ = false, ran_ = false;
  unsigned NUM_EXPERTS = EXPERTS, MOE_CACHE_EXPERTS = 16;
  void beforePrefillBlock(unsigned int from, unsigned int to);
  void beginRun() {
    RUN_ENTRY
    ran_ = true;
  }
};
using Model = MODEL_NAME;
'''

arrays = r'''

std::vector<std::vector<int>> active, recency;
std::vector<std::pair<int, std::vector<int>>> future_routes;
void readRoutes(const char *path, const char *future_path) {
  std::ifstream input(path);
  assert(input);
  std::string line;
  while (std::getline(input, line)) {
    const auto p = line.find('|'), q = line.find('|', p + 1);
    assert(p != std::string::npos && q != std::string::npos);
    std::istringstream h(line.substr(0, p)), a(line.substr(p + 1, q - p - 1)),
      r(line.substr(q + 1));
    int layer, tokens, e;
    assert(bool(h >> layer >> tokens));
    assert(layer == int(active.size()) && tokens > 1);
    std::vector<int> av, rv;
    while (a >> e) {
      assert(e >= 0 && e < EXPERTS);
      av.push_back(e);
    }
    while (r >> e) {
      assert(e >= 0 && e < EXPERTS);
      rv.push_back(e);
    }
    assert(!av.empty() && !rv.empty());
    std::reverse(rv.begin(), rv.end());
    active.push_back(std::move(av));
    recency.push_back(std::move(rv));
  }
  assert(active.size() == LAYERS);
  if (future_path) {
    std::ifstream future(future_path);
    assert(future);
    while (std::getline(future, line)) {
      const auto p = line.find('|'), q = line.find('|', p + 1);
      assert(p != std::string::npos && q != std::string::npos);
      std::istringstream h(line.substr(0, p)), a(line.substr(p + 1, q - p - 1));
      int layer, token, e;
      assert(bool(h >> layer >> token));
      assert(layer >= 0 && layer < LAYERS && token == 1);
      std::vector<int> keys;
      while (a >> e) {
        assert(e >= 0 && e < EXPERTS);
        keys.push_back(e);
      }
      assert(!keys.empty());
      future_routes.emplace_back(layer, std::move(keys));
    }
  } else {
    // Deterministic probes after the actual prefill expose hidden LRFU scores.
    // These are a cache-state test, not a claim about device decode misses.
    for (int pass = 0; pass < 4; ++pass)
      for (int layer = 0; layer < LAYERS; ++layer) {
        std::vector<int> keys;
        for (int e = pass; e < EXPERTS; e += 8)
          keys.push_back(e);
        future_routes.emplace_back(layer, std::move(keys));
      }
  }
}
'''

checks = r'''

struct Result {
  std::vector<std::vector<int>> reads, registrations;
  std::vector<int> evictions, final_order, future_misses, future_evictions,
    future_order;
  int misses;
};
Result run(bool early) {
  Fake ops;
  seed(ops);
  if (early) {
    assert(BEGIN(&ops) == EARLY_READS);
    assert(g_expert_lru.size() == 128 && g_expert_lru.capacity() == CAPACITY);
    assert(ops.evictions.empty() && ops.reads.size() == EARLY_BATCHES &&
           ops.registrations.empty());
    const auto before = g_expert_lru.order();
    assert(BEGIN(&ops) == 0);
    assert(g_expert_lru.order() == before && ops.reads.size() == EARLY_BATCHES);
  }
  for (int l = 0; l < LAYERS; ++l) {
    forwardLayer(ops, l, active[l], recency[l]);
    if (!l) {
      assert(ops.end_calls == 0);
      assert(g_prefetch_early_owner == nullptr);
    }
    assert(g_expert_lru.size() <= CAPACITY &&
           g_expert_lru.capacity() == CAPACITY);
  }
  assert(g_prefetch_batches.empty() && g_prefetch_inflight.empty() &&
         ops.batches.empty());
  assert(BEGIN(&ops) == 0);
  assert(ops.demand_reads == 0);
  Result result{ops.reads,
                ops.registrations,
                ops.evictions,
                ids(g_expert_lru.order()),
                {},
                {},
                {},
                ops.demand_reads};
  ops.evictions.clear();
  for (const auto &row : future_routes) {
    std::vector<const void *> keys;
    for (int e : row.second)
      keys.push_back(&identities[row.first * EXPERTS + e]);
    g_expert_lru.acquire(
      keys,
      [&](const void *k) {
        result.future_misses.push_back(id(k));
        ops.resident.insert(k);
      },
      [&](const void *k) { ops.evict(k); });
    g_expert_lru.refresh(keys);
  }
  result.future_evictions = ops.evictions;
  result.future_order = ids(g_expert_lru.order());
  return result;
}
void checkGuards() {
  {
    Fake a, b;
    seed(a);
    assert(BEGIN(nullptr) == 0 && BEGIN(&b) == 0);
    assert(a.reads.empty() && b.reads.empty());
  }
  {
    Fake a;
    seed(a, 64);
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a, 128, 22);
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_expert_layers[8][0].w_bits = 2;
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_expert_layers.pop_back();
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_expert_layers[8].pop_back();
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_handed_expert_layers.insert(&identities[0]);
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_prefetch_inflight.insert(&identities[200]);
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_prefetch_batches.emplace_back(1, 1);
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    g_prefetch_next = 1;
    assert(BEGIN(&a) == 0);
  }
  {
    Fake a;
    seed(a);
    a.fail_begin = 0;
    a.refuse = true;
    assert(BEGIN(&a) == 0);
    assert(g_prefetch_next == 0 &&
           g_expert_lru.room(g_expert_lru.order()) == CAPACITY - 128);
  }
  {
    Fake a;
    seed(a);
    a.fail_begin = 0;
    bool threw = false;
    try {
      BEGIN(&a);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && a.reads.empty() && g_prefetch_batches.empty());
    assert(g_expert_lru.room(g_expert_lru.order()) == CAPACITY - 128);
    a.stopWorkers();
    resetLfm2HtpExpertPool();
  }
  {
    Fake a;
    seed(a);
    a.fail_begin = 1;
    bool threw = false;
    try {
      BEGIN(&a);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw && a.reads.size() == 1 && g_prefetch_batches.size() == 1);
    assert(g_expert_lru.room(g_expert_lru.order()) == CAPACITY - 128 - EXPERTS);
    a.stopWorkers();
    resetLfm2HtpExpertPool();
    assert(g_prefetch_batches.empty() && !g_prefetch_early_owner &&
           !g_expert_owner);
  }
  {
    Fake a;
    seed(a);
    a.fail_begin = 1;
    a.refuse = true;
    assert(BEGIN(&a) == EXPERTS);
    a.fail_begin = -1;
    for (int l = 0; l < LAYERS; ++l)
      forwardLayer(a, l, active[l], recency[l]);
    assert(a.demand_reads == 0 && g_prefetch_batches.empty());
  }
  {
    Fake a, b;
    seed(a);
    assert(BEGIN(&a) == EARLY_READS);
    bool threw = false;
    try {
      forwardLayer(b, 0, active[0], recency[0]);
    } catch (const std::logic_error &) {
      threw = true;
    }
    assert(threw && a.end_calls == 0 && b.end_calls == 0);
    a.stopWorkers();
    resetLfm2HtpExpertPool();
  }
  {
    Fake a;
    seed(a);
    assert(BEGIN(&a) == EARLY_READS);
    a.stopWorkers();
    resetLfm2HtpExpertPool();
    assert(g_expert_lru.size() == 0 && g_expert_lru.capacity() == 0 &&
           !g_expert_owner && !g_prefetch_early_owner);
    Fake b;
    seed(b);
    assert(BEGIN(&b) == EARLY_READS);
    b.stopWorkers();
    resetLfm2HtpExpertPool();
  }
}
void checkModelHook() {
  {
    Fake a;
    seed(a);
    unsetenv(ENV_NAME);
    Model model;
    model.beginRun();
    model.beforePrefillBlock(0, 1024);
    assert(a.reads.empty());
  }
  {
    Fake a;
    seed(a);
    setenv(ENV_NAME, "0", 1);
    Model model;
    model.beginRun();
    model.beforePrefillBlock(0, 1024);
    assert(a.reads.empty());
  }
  {
    Fake a;
    seed(a);
    setenv(ENV_NAME, "1", 1);
    Model model;
    model.beginRun();
    model.beforePrefillBlock(0, 1);
    model.beforePrefillBlock(1, 1024);
    assert(a.reads.empty());
    model.beforePrefillBlock(0, 1024);
    assert(a.reads.size() == EARLY_BATCHES);
    model.beforePrefillBlock(0, 1024);
    assert(a.reads.size() == EARLY_BATCHES);
  }
  {
    Fake a;
    seed(a);
    Model model;
    model.beginRun();
    model.prepared_ = false;
    model.beforePrefillBlock(0, 1024);
    assert(a.reads.empty());
  }
  EXTRA_MODEL_GUARDS
}
} // namespace causallm
int main(int argc, char **argv) {
  assert(argc == 2 || argc == 3);
  using namespace causallm;
  readRoutes(argv[1], argc == 3 ? argv[2] : nullptr);
  const auto baseline = run(false), early = run(true);
  assert(baseline.reads == early.reads &&
         baseline.registrations == early.registrations);
  assert(baseline.evictions == early.evictions &&
         baseline.final_order == early.final_order);
  assert(baseline.future_misses == early.future_misses &&
         baseline.future_evictions == early.future_evictions &&
         baseline.future_order == early.future_order);
  assert(baseline.misses == early.misses);
  size_t count = 0;
  for (const auto &batch : early.reads)
    count += batch.size();
  assert(count == LAYERS * EXPERTS - 128);
  checkGuards();
  checkModelHook();
  std::cout << "PASS: actual " << LAYERS << "-layer route, " << count
            << " identical reads, " << baseline.registrations.size()
            << " identical registration groups, " << baseline.evictions.size()
            << " identical prefill evictions, identical final LRFU state, zero "
               "demand misses; "
            << future_routes.size() << " future probes with "
            << baseline.future_misses.size() << " identical misses/evictions; C"
            << CAPACITY
            << "/PRELOAD128, owner, old-pass, fallback, queue failure and "
               "teardown guards\n";
}
'''

restart_checks = r'''

// The first run either fails during kickoff or is interrupted after kickoff
// and before layer zero. A second run must retire the old early marker and
// drain exactly those owned batches through the ordinary layer-zero path.
for (bool fail_queue : {false, true}) {
  Fake a, other;
  seed(a);
  Model model;
  model.beginRun();
  if (fail_queue)
    a.fail_begin = 1;
  bool threw = false;
  try {
    model.beforePrefillBlock(0, 1024);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  assert(threw == fail_queue);
  const size_t old_batches = a.batches.size();
  assert(old_batches == (fail_queue ? 1u : size_t(EARLY_BATCHES)));
  assert(g_prefetch_early_owner == &a);
  retireHtpExpertPrefillKickoff(&other);
  retireHtpExpertPrefillKickoff(nullptr);
  assert(g_prefetch_early_owner == &a);
  a.fail_begin = -1;
  model.beginRun();
  assert(!g_prefetch_early_owner && a.batches.size() == old_batches);
  model.beforePrefillBlock(0, 1024);
  assert(a.batches.size() == old_batches);
  a.old_pass = true;
  forwardLayer(a, 0, active[0], recency[0]);
  a.old_pass = false;
  assert(size_t(a.end_calls) == old_batches);
  for (int l = 1; l < LAYERS; ++l)
    forwardLayer(a, l, active[l], recency[l]);
  assert(a.demand_reads == 0 && g_prefetch_batches.empty());
  assert(g_prefetch_inflight.empty() && a.batches.empty());
  assert(g_expert_lru.capacity() == CAPACITY);
}

{
  Fake a;
  seed(a);
  Model model;
  model.beginRun();
  model.beginRun();
  model.beforePrefillBlock(0, 1024);
  assert(a.reads.empty());
}
'''

# Only the new lifecycle prefix is under test here. The upstream backend
# attention/conv/KV reset itself is covered by model/device checks.
run_block = function(model_text, '  if (ran_) {')
run_entry = run_block[:run_block.index('nntrainer::htp_begin_prompt();')]
run_entry += '}\n'
model_stub = model_stub.replace('MODEL_NAME', model_name)
model_stub = model_stub.replace('RUN_ENTRY', run_entry)
checks = checks.replace('EXTRA_MODEL_GUARDS', restart_checks)
checks = checks.replace('BEGIN', 'beginLfm2HtpExpertPrefill')
checks = checks.replace('ENV_NAME', '"NNTR_LFM2_PREFILL_EARLY"')
source = (header + config + helper + fixture + collect + stage + queue + tail +
          model_stub + hook + arrays + checks)
compiler = os.environ.get('CXX', 'g++')
flags = ['-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-pthread',
         '-I' + str(app / 'lfm2_moe')]
if args.sanitize:
    flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
with tempfile.TemporaryDirectory(prefix='nntr-early-prefill-') as temporary:
    directory = args.output_dir.resolve() if args.output_dir else Path(temporary)
    directory.mkdir(parents=True, exist_ok=True)
    cpp = directory / 'early_prefill_check.cpp'
    binary = directory / 'early_prefill_check'
    cpp.write_text(source)
    subprocess.run([compiler, *flags, str(cpp), '-o', str(binary)], check=True)
    command = [str(binary), str(args.routes.resolve())]
    if args.decode_routes:
        command.append(str(args.decode_routes.resolve()))
    env = dict(os.environ)
    # Match the measured full-horizon default; run cache-policy sweeps by
    # explicitly setting NNTR_MOE_LRFU in the parent environment.
    env.pop('NNTR_MOE_PREFETCH', None)
    env.pop('NNTR_MOE_SPLIT', None)
    subprocess.run(command, env=env, check=True)
