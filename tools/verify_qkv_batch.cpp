// SPDX-License-Identifier: Apache-2.0
/**
 * @file   verify_qkv_batch.cpp
 * @brief  Standalone, host-only correctness check for QKVLayer.
 *
 * Cheap isolation test recommended in
 * docs/backend_guide/HEXAGON_NPU_OBSERVATION_LOG.md S25: build a tiny model
 * using QKVLayer via the ordinary string-based "input_layers=name(idx)"
 * connection syntax (not CausalLM's symbolic Tensor::output() API), with
 * known deterministic weights, and diff against 3 separate fully_connected
 * layers sharing the same input and identical weight values. FP32 only - no
 * Hexagon, no Q4_0, no Android round trip needed, since the stash's own
 * bisection already showed the garbage output reproduces on the plain FP32
 * model.
 *
 * Exit code 0 = outputs match (bug is NOT in QKVLayer's core finalize/
 * forwarding/weight-request logic - look at the CausalLM symbolic wiring or
 * incremental-forwarding usage instead). Nonzero = mismatch, and this
 * isolates the bug to QKVLayer itself, independent of CausalLM's model
 * files.
 *
 * Standalone like tools/nntr_htp_bridge_check.cpp - deliberately not wired
 * into meson, since it links against a host build's libnntrainer and is a
 * one-shot diagnostic rather than part of the test suite. Build with:
 *
 *   meson setup build-verify -Denable-fp16=false -Denable-transformer=true \
 *       -Denable-test=false -Dthread-backend=omp
 *   ninja -C build-verify nntrainer/libnntrainer.so
 *   c++ -std=c++17 -O0 -g $(python3 -c "import json;print(' '.join(
 *       f for e in json.load(open('build-verify/compile_commands.json'))
 *       if e['file'].endswith('fc_layer.cpp')
 *       for f in e['command'].split() if f.startswith(('-I','-D'))))") \
 *     tools/verify_qkv_batch.cpp -Lbuild-verify/nntrainer -lnntrainer \
 *     -Wl,-rpath,$PWD/build-verify/nntrainer -pthread -o /tmp/verify_qkv_batch
 */

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer_node.h>
#include <neuralnet.h>
#include <tensor.h>

using LayerRep = std::pair<std::string, std::vector<std::string>>;

static nntrainer::GraphRepresentation
makeGraph(const std::vector<LayerRep> &reps) {
  auto &eg = nntrainer::Engine::Global();
  nntrainer::GraphRepresentation g;
  for (auto &r : reps) {
    std::shared_ptr<nntrainer::LayerNode> layer = nntrainer::createLayerNode(
      eg.createLayerObject(r.first, r.second), r.second);
    g.push_back(layer);
  }
  return g;
}

static void fillDeterministic(nntrainer::Tensor &t, int seed) {
  float *data = t.getData<float>();
  size_t n = t.size();
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::sin(0.137f * (float)(i + (size_t)seed * 97));
  }
}

static nntrainer::Tensor &weightOf(nntrainer::NeuralNetwork &nn,
                                  const char *layer_name, unsigned int idx) {
  std::shared_ptr<ml::train::Layer> l;
  int ret = nn.getLayer(layer_name, &l);
  if (ret != ML_ERROR_NONE) {
    throw std::runtime_error(std::string("layer not found: ") + layer_name);
  }
  auto node = std::dynamic_pointer_cast<nntrainer::LayerNode>(l);
  if (!node) {
    throw std::runtime_error(std::string("not a LayerNode: ") + layer_name);
  }
  return node->getRunContext().getWeight(idx);
}

static float maxAbsDiff(const nntrainer::Tensor &a, const nntrainer::Tensor &b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("size mismatch in comparison");
  }
  float max_diff = 0.0f;
  const float *da = a.getData<float>();
  const float *db = b.getData<float>();
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(da[i] - db[i]));
  }
  return max_diff;
}

int main() {
  constexpr unsigned int SEQ = 4;   // M: a few "tokens" (prefill-like, M > 1)
  constexpr unsigned int HID = 8;   // K: hidden dim
  constexpr unsigned int UNIT = 8;  // Q/K/V unit size (kept equal, small)

  const std::string input_shape =
    "input_shape=1:" + std::to_string(SEQ) + ":" + std::to_string(HID);

  // Model A: baseline - 3 independent fully_connected layers sharing input,
  // no bias (QKVLayer has no bias weight, so this is the fair comparison).
  nntrainer::NeuralNetwork nnA;
  nnA.setProperty({"batch_size=1"});
  {
    auto graph = makeGraph({
      {"input", {"name=input", input_shape}},
      {"fully_connected",
       {"name=fc_q", "input_layers=input", "unit=" + std::to_string(UNIT),
        "disable_bias=true"}},
      {"fully_connected",
       {"name=fc_k", "input_layers=input", "unit=" + std::to_string(UNIT),
        "disable_bias=true"}},
      {"fully_connected",
       {"name=fc_v", "input_layers=input", "unit=" + std::to_string(UNIT),
        "disable_bias=true"}},
      {"activation", {"name=q_out", "activation=none", "input_layers=fc_q"}},
      {"activation", {"name=k_out", "activation=none", "input_layers=fc_k"}},
      {"activation", {"name=v_out", "activation=none", "input_layers=fc_v"}},
    });
    for (auto &node : graph) {
      nnA.addLayer(node);
    }
  }

  // Model B: QKVLayer, one dispatch for all three.
  nntrainer::NeuralNetwork nnB;
  nnB.setProperty({"batch_size=1"});
  {
    auto graph = makeGraph({
      {"input", {"name=input", input_shape}},
      {"qkv_layer",
       {"name=qkv", "input_layers=input", "q_unit=" + std::to_string(UNIT),
        "k_unit=" + std::to_string(UNIT), "v_unit=" + std::to_string(UNIT)}},
      {"activation", {"name=q_out", "activation=none", "input_layers=qkv(0)"}},
      {"activation", {"name=k_out", "activation=none", "input_layers=qkv(1)"}},
      {"activation", {"name=v_out", "activation=none", "input_layers=qkv(2)"}},
    });
    for (auto &node : graph) {
      nnB.addLayer(node);
    }
  }

  printf("compiling model A (3x fully_connected)...\n");
  if (nnA.compile(ml::train::ExecutionMode::INFERENCE)) {
    fprintf(stderr, "model A compile failed\n");
    return 1;
  }
  printf("initializing model A...\n");
  if (nnA.initialize()) {
    fprintf(stderr, "model A initialize failed\n");
    return 1;
  }

  printf("compiling model B (qkv_layer)...\n");
  if (nnB.compile(ml::train::ExecutionMode::INFERENCE)) {
    fprintf(stderr, "model B compile failed\n");
    return 1;
  }
  printf("initializing model B...\n");
  if (nnB.initialize()) {
    fprintf(stderr, "model B initialize failed\n");
    return 1;
  }

  // Copy identical deterministic weight values into both models.
  nntrainer::Tensor &wq_a = weightOf(nnA, "fc_q", 0);
  nntrainer::Tensor &wk_a = weightOf(nnA, "fc_k", 0);
  nntrainer::Tensor &wv_a = weightOf(nnA, "fc_v", 0);
  nntrainer::Tensor &wq_b = weightOf(nnB, "qkv", 0);
  nntrainer::Tensor &wk_b = weightOf(nnB, "qkv", 1);
  nntrainer::Tensor &wv_b = weightOf(nnB, "qkv", 2);

  if (wq_a.size() != wq_b.size() || wk_a.size() != wk_b.size() ||
      wv_a.size() != wv_b.size()) {
    fprintf(stderr,
            "FATAL: weight shapes differ between fc_* and qkv_layer - "
            "wq_a=%zu wq_b=%zu wk_a=%zu wk_b=%zu wv_a=%zu wv_b=%zu\n",
            wq_a.size(), wq_b.size(), wk_a.size(), wk_b.size(), wv_a.size(),
            wv_b.size());
    return 1;
  }

  fillDeterministic(wq_a, 1);
  fillDeterministic(wk_a, 2);
  fillDeterministic(wv_a, 3);
  // copy same bytes into B's weights (not re-derive - must be bit-identical)
  std::copy(wq_a.getData<float>(), wq_a.getData<float>() + wq_a.size(),
            wq_b.getData<float>());
  std::copy(wk_a.getData<float>(), wk_a.getData<float>() + wk_a.size(),
            wk_b.getData<float>());
  std::copy(wv_a.getData<float>(), wv_a.getData<float>() + wv_a.size(),
            wv_b.getData<float>());

  // Identical input activation for both models.
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, SEQ, HID,
                        nntrainer::TensorDim::TensorType(
                          nntrainer::Tformat::NCHW,
                          nntrainer::TensorDim::DataType::FP32)));
  fillDeterministic(input, 42);

  printf("running inference on model A...\n");
  nntrainer::sharedConstTensors outA =
    nnA.inference({MAKE_SHARED_TENSOR(input)}, false);
  printf("running inference on model B...\n");
  nntrainer::sharedConstTensors outB =
    nnB.inference({MAKE_SHARED_TENSOR(input)}, false);

  if (outA.size() != 3 || outB.size() != 3) {
    fprintf(stderr, "FATAL: expected 3 outputs each, got A=%zu B=%zu\n",
            outA.size(), outB.size());
    return 1;
  }

  const char *names[3] = {"Q", "K", "V"};
  bool all_match = true;
  for (int i = 0; i < 3; ++i) {
    float diff = maxAbsDiff(*outA[i], *outB[i]);
    printf("%s: max_abs_diff = %g (size a=%zu b=%zu)\n", names[i], diff,
          outA[i]->size(), outB[i]->size());
    if (diff > 1e-4f) {
      all_match = false;
    }
  }

  if (all_match) {
    printf("PASS: qkv_layer output matches 3 separate fully_connected "
          "layers.\n");
    return 0;
  } else {
    printf("FAIL: qkv_layer output diverges from 3 separate "
          "fully_connected layers - bug is in QKVLayer itself.\n");
    return 2;
  }
}
