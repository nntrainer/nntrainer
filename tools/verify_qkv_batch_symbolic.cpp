// SPDX-License-Identifier: Apache-2.0
/**
 * @file   verify_qkv_batch_symbolic.cpp
 * @brief  Same check as verify_qkv_batch.cpp, but wired through the
 *         symbolic ml::train::Tensor/LayerHandle API - the exact mechanism
 *         qwen3_causallm.cpp uses (Tensor qkv_out = wqkv(query); Tensor q =
 *         qkv_out.output(0); ...) - instead of the plain string
 *         "input_layers=name(idx)" syntax verify_qkv_batch.cpp used.
 *
 * Also uses asymmetric Q/K/V unit sizes (mirroring GQA: k_unit/v_unit <
 * q_unit) and weight_initializer=ones, matching qwen3_causallm.cpp's actual
 * createAttention() call exactly - verify_qkv_batch.cpp used equal units,
 * which would silently mask a bug tied to the asymmetric-size path (e.g. if
 * finalize()'s in-place weight_dim.width() mutation between Q/K/V requests
 * were captured by reference rather than by value - checked and ruled out
 * statically, but this test covers the scenario empirically too).
 *
 * Build exactly as verify_qkv_batch.cpp (see its header), plus
 * -Lbuild-verify/api/ccapi -lccapi-nntrainer and a matching -Wl,-rpath, since
 * this one uses the ccapi symbolic Tensor/LayerHandle/Model API.
 */

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <app_context.h>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <tensor_api.h>
#include <tensor.h>

using ml::train::createLayer;
using ml::train::createModel;
using ml::train::ExecutionMode;
using ml::train::LayerHandle;
using ml::train::ModelType;
using ml::train::Tensor;

static std::string withKey(const std::string &k, const std::string &v) {
  return k + "=" + v;
}

static void fillDeterministic(float *data, size_t n, int seed) {
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::sin(0.137f * (float)(i + (size_t)seed * 97));
  }
}

static nntrainer::Tensor &weightOf(std::unique_ptr<ml::train::Model> &model,
                                  const char *layer_name, unsigned int idx) {
  std::shared_ptr<ml::train::Layer> l;
  int ret = model->getLayer(layer_name, &l);
  if (ret != ML_ERROR_NONE) {
    throw std::runtime_error(std::string("layer not found: ") + layer_name);
  }
  auto node = std::dynamic_pointer_cast<nntrainer::LayerNode>(l);
  if (!node) {
    throw std::runtime_error(std::string("not a LayerNode: ") + layer_name);
  }
  return node->getRunContext().getWeight(idx);
}

int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  constexpr unsigned int SEQ = 4;
  constexpr unsigned int HID = 8;
  constexpr unsigned int Q_UNIT = 8; // asymmetric, mirrors GQA q vs k/v split
  constexpr unsigned int K_UNIT = 4;
  constexpr unsigned int V_UNIT = 4;

  // ---- Model A: 3 separate fully_connected, symbolic wiring ----
  auto modelA = createModel(ModelType::NEURAL_NET);
  modelA->setProperty({"batch_size=1"});

  Tensor xA({1, 1, SEQ, HID}, "inputA");

  LayerHandle fc_q(createLayer(
    "fully_connected", {withKey("name", "fc_q"), withKey("unit", std::to_string(Q_UNIT)),
                        withKey("disable_bias", "true"),
                        withKey("weight_initializer", "ones")}));
  LayerHandle fc_k(createLayer(
    "fully_connected", {withKey("name", "fc_k"), withKey("unit", std::to_string(K_UNIT)),
                        withKey("disable_bias", "true"),
                        withKey("weight_initializer", "ones")}));
  LayerHandle fc_v(createLayer(
    "fully_connected", {withKey("name", "fc_v"), withKey("unit", std::to_string(V_UNIT)),
                        withKey("disable_bias", "true"),
                        withKey("weight_initializer", "ones")}));

  Tensor qA = fc_q(xA);
  Tensor kA = fc_k(xA);
  Tensor vA = fc_v(xA);

  LayerHandle qoutA(createLayer(
    "activation", {withKey("name", "q_out"), withKey("activation", "none")}));
  LayerHandle koutA(createLayer(
    "activation", {withKey("name", "k_out"), withKey("activation", "none")}));
  LayerHandle voutA(createLayer(
    "activation", {withKey("name", "v_out"), withKey("activation", "none")}));

  Tensor qoA = qoutA(qA);
  Tensor koA = koutA(kA);
  Tensor voA = voutA(vA);

  std::vector<Tensor> outsA = {qoA, koA, voA};
  printf("compiling model A (symbolic, 3x fully_connected)...\n");
  if (modelA->compile(xA, outsA, ExecutionMode::INFERENCE)) {
    fprintf(stderr, "model A compile failed\n");
    return 1;
  }

  // ---- Model B: qkv_layer, symbolic wiring via .output(idx) ----
  auto modelB = createModel(ModelType::NEURAL_NET);
  modelB->setProperty({"batch_size=1"});

  Tensor xB({1, 1, SEQ, HID}, "inputB");

  LayerHandle wqkv(createLayer(
    "qkv_layer", {withKey("name", "qkv"), withKey("q_unit", std::to_string(Q_UNIT)),
                 withKey("k_unit", std::to_string(K_UNIT)),
                 withKey("v_unit", std::to_string(V_UNIT)),
                 withKey("disable_bias", "true"),
                 withKey("weight_initializer", "ones")}));

  Tensor qkv_out = wqkv(xB);
  Tensor qB = qkv_out.output(0);
  Tensor kB = qkv_out.output(1);
  Tensor vB = qkv_out.output(2);

  LayerHandle qoutB(createLayer(
    "activation", {withKey("name", "q_out"), withKey("activation", "none")}));
  LayerHandle koutB(createLayer(
    "activation", {withKey("name", "k_out"), withKey("activation", "none")}));
  LayerHandle voutB(createLayer(
    "activation", {withKey("name", "v_out"), withKey("activation", "none")}));

  Tensor qoB = qoutB(qB);
  Tensor koB = koutB(kB);
  Tensor voB = voutB(vB);

  std::vector<Tensor> outsB = {qoB, koB, voB};
  printf("compiling model B (symbolic, qkv_layer)...\n");
  if (modelB->compile(xB, outsB, ExecutionMode::INFERENCE)) {
    fprintf(stderr, "model B compile failed\n");
    return 1;
  }

  // ---- Copy identical deterministic weights into both ----
  nntrainer::Tensor &wq_a = weightOf(modelA, "fc_q", 0);
  nntrainer::Tensor &wk_a = weightOf(modelA, "fc_k", 0);
  nntrainer::Tensor &wv_a = weightOf(modelA, "fc_v", 0);
  nntrainer::Tensor &wq_b = weightOf(modelB, "qkv", 0);
  nntrainer::Tensor &wk_b = weightOf(modelB, "qkv", 1);
  nntrainer::Tensor &wv_b = weightOf(modelB, "qkv", 2);

  if (wq_a.size() != wq_b.size() || wk_a.size() != wk_b.size() ||
      wv_a.size() != wv_b.size()) {
    fprintf(stderr,
            "FATAL: weight shapes differ - wq_a=%zu wq_b=%zu wk_a=%zu "
            "wk_b=%zu wv_a=%zu wv_b=%zu\n",
            wq_a.size(), wq_b.size(), wk_a.size(), wk_b.size(), wv_a.size(),
            wv_b.size());
    return 1;
  }

  fillDeterministic(wq_a.getData<float>(), wq_a.size(), 1);
  fillDeterministic(wk_a.getData<float>(), wk_a.size(), 2);
  fillDeterministic(wv_a.getData<float>(), wv_a.size(), 3);
  std::copy(wq_a.getData<float>(), wq_a.getData<float>() + wq_a.size(),
            wq_b.getData<float>());
  std::copy(wk_a.getData<float>(), wk_a.getData<float>() + wk_a.size(),
            wk_b.getData<float>());
  std::copy(wv_a.getData<float>(), wv_a.getData<float>() + wv_a.size(),
            wv_b.getData<float>());

  // ---- Identical input activation ----
  std::vector<float> input_data(SEQ * HID);
  fillDeterministic(input_data.data(), input_data.size(), 42);

  printf("running inference on model A...\n");
  std::vector<float *> outA =
    modelA->inference(1, {input_data.data()});
  printf("running inference on model B...\n");
  std::vector<float *> outB =
    modelB->inference(1, {input_data.data()});

  if (outA.size() != 3 || outB.size() != 3) {
    fprintf(stderr, "FATAL: expected 3 outputs each, got A=%zu B=%zu\n",
            outA.size(), outB.size());
    return 1;
  }

  // getOutputDimension() requires a requireLabel()-true node (a loss layer),
  // which neither inference-only graph has - compute expected sizes directly
  // instead of calling it.
  const size_t sizes[3] = {(size_t)SEQ * Q_UNIT, (size_t)SEQ * K_UNIT,
                           (size_t)SEQ * V_UNIT};

  const char *names[3] = {"Q", "K", "V"};
  bool all_match = true;
  for (int i = 0; i < 3; ++i) {
    size_t n = sizes[i];
    float max_diff = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      max_diff = std::max(max_diff, std::fabs(outA[i][j] - outB[i][j]));
    }
    printf("%s: max_abs_diff = %g (size=%zu)\n", names[i], max_diff, n);
    if (max_diff > 1e-4f) {
      all_match = false;
    }
  }

  if (all_match) {
    printf("PASS: symbolic-wired qkv_layer matches symbolic-wired 3x "
          "fully_connected.\n");
    return 0;
  } else {
    printf("FAIL: symbolic-wired qkv_layer diverges - bug is in the "
          "symbolic Tensor::output(idx) wiring path or asymmetric-unit "
          "handling.\n");
    return 2;
  }
}
