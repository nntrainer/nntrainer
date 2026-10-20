// SPDX-License-Identifier: Apache-2.0
/**
 * @file static_encoding_check.cpp
 * @brief Validate static grids, logical-to-physical mapping and QDQ ties.
 * @author Codex <noreply@openai.com>
 * @bug See the repository issue tracker for known issues.
 */
#include <cassert>
#include <fstream>
#include <gemma4_htp_static_quant.h>
#include <htp_static_quant.h>
#include <iostream>

int main(int argc, char **argv) {
  using causallm::parseHtpStaticLinear;
  using Json = nlohmann::json;
  const Json grid = {
    {"bitwidth", 8}, {"dtype", "int"}, {"scale", 0.5}, {"offset", -129}};
  Json source = {{"sites",
                  {{"model.layers.2.experts.gate_up_proj/in", grid},
                   {"model.layers.2.self_attn.o_proj/in", grid},
                   {"lm_head/in", grid}}}};
  auto encoded = parseHtpStaticLinear(source, 3);
  assert(encoded.size() == 5);
  assert(encoded.at("layer2_sparse_moe:expert_gate_up_2").zero_point == 129);
  assert(encoded.at("layer2_post_attention_norm:weight").scale == 0.5f);
  const Json alias = {{"layer2_post_attention_norm:proj", {0.5, 129}}};
  assert(
    parseHtpStaticLinear(alias, 3).count("layer2_post_attention_norm:weight"));
  auto rejects = [](const Json &v) {
    bool threw = false;
    try {
      (void)parseHtpStaticLinear(v, 3);
    } catch (const std::exception &) {
      threw = true;
    }
    assert(threw);
  };
  for (const Json &bad :
       {Json::array(), Json::object(), Json{{"lm_head/in", {0.5, 129}}},
        Json{{"output_of_causallm:weight", {0, 0}}},
        Json{{"output_of_causallm:weight", {1e-100, 0}}},
        Json{{"output_of_causallm:weight", {1e100, 0}}},
        Json{{"output_of_causallm:weight", {1e-38, 0}}},
        Json{{"output_of_causallm:weight", {1e38, 0}}},
        Json{{"output_of_causallm:weight",
              {std::numeric_limits<double>::quiet_NaN(), 0}}},
        Json{{"output_of_causallm:weight",
              {std::numeric_limits<double>::infinity(), 0}}},
        Json{{"output_of_causallm:weight",
              {0.5, std::numeric_limits<double>::quiet_NaN()}}},
        Json{{"output_of_causallm:weight", {0.5, 2.5}}},
        Json{{"output_of_causallm:weight", {0.5, 256}}},
        Json{{"output_of_causallm:weight", {0.5, -1}}},
        Json{{"layer2_post_attention_norm:proj", {0.5, 129}},
             {"layer2_post_attention_norm:weight", {0.5, 129}}}})
    rejects(bad);
  source["sites"]["lm_head/in"]["bitwidth"] = 16;
  rejects(source);
  // Offset parity must not change ties: round x/scale, then add zero point.
  assert(htp_static_qdq(0.25f, 0.5f, 129) == 0.0f);
  assert(htp_static_qdq(0.75f, 0.5f, 129) == 1.0f);
  assert(htp_static_qdq(-0.75f, 0.5f, 129) == -1.0f);
  assert(htp_static_qdq(1000.0f, 0.5f, 129) == 63.0f);
  assert(htp_static_qdq(-1000.0f, 0.5f, 129) == -64.5f);
  if (argc == 3) {
    std::ifstream physical(argv[1]), logical(argv[2]);
    auto a = parseHtpStaticLinear(Json::parse(physical), 128);
    auto b = parseHtpStaticLinear(Json::parse(logical), 128);
    assert(a.size() == b.size());
    for (const auto &[name, q] : a) {
      assert(b.at(name).scale == q.scale);
      assert(b.at(name).zero_point == q.zero_point);
      assert(causallm::htpStaticLinearWeight(name));
    }
    std::cout << "Matched " << a.size() << " checkpoint weight inputs\n";
  }
  std::cout << "STATIC ENCODING CHECK OK\n";
}
