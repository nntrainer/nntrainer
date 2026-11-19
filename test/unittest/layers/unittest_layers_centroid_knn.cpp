// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_layers_centroid_knn.cpp
 * @brief Check classification when some classes have no observed samples.
 * @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
 * @bug No known bugs.
 */

#include <gtest/gtest.h>

#include <centroid_knn.h>
#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

namespace {

class CentroidKNNTest : public ::testing::TestWithParam<unsigned int> {};

/** @brief Run the native layer with known centroids and sample counts. */
void checkClassification(unsigned int batch, unsigned int observed,
                         bool exact_match, bool all_observed) {
  using namespace nntrainer;
  CentroidKNN layer;
  layer.setProperty({"num_class=3"});
  TensorDim input_dim(batch, 1, 1, 2);
  InitLayerContext init({input_dim}, {true}, false, "centroid_knn");
  layer.finalize(init);
  ASSERT_EQ(init.getWeightsSpec().size(), 2U);

  Weight centroids(std::get<0>(init.getWeightsSpec()[0]), Initializer::ZEROS,
                   WeightRegularizer::NONE, 1.0f, 0.0f, 0.0f, false, true,
                   "centroids");
  Weight counts(std::get<0>(init.getWeightsSpec()[1]), Initializer::ZEROS,
                WeightRegularizer::NONE, 1.0f, 0.0f, 0.0f, false, true,
                "counts");
  Var_Grad input(input_dim, Initializer::ZEROS, false, true, "input");
  TensorDim output_dim = init.getOutSpecs()[0].variable_spec.dim;
  output_dim.batch(batch);
  Var_Grad output(output_dim, Initializer::NONE, false, true, "output");

  for (unsigned int cls = 0; cls < 3; ++cls) {
    if (all_observed) {
      centroids.getVariableRef().getData<float>()[2 * cls] = 3.0f * cls;
      centroids.getVariableRef().getData<float>()[2 * cls + 1] = 4.0f * cls;
    }
    if (all_observed || cls == observed)
      counts.getVariableRef().getData<float>()[cls] = 1.0f;
  }
  for (unsigned int b = 0; b < batch; ++b) {
    input.getVariableRef().getData<float>()[2 * b] = exact_match ? 0.0f : 3.0f;
    input.getVariableRef().getData<float>()[2 * b + 1] =
      exact_match ? 0.0f : 4.0f;
  }
  Tensor saved_centroids = centroids.getVariableRef().clone();
  Tensor saved_counts = counts.getVariableRef().clone();
  RunLayerContext run("centroid_knn", false, 0.0f, false, 1.0f, nullptr, false,
                      {&centroids, &counts}, {&input}, {&output}, {});
  layer.forwarding(run, false);

  const auto predicted = output.getVariableRef().argmax();
  ASSERT_EQ(predicted.size(), batch);
  for (unsigned int b = 0; b < batch; ++b) {
    EXPECT_EQ(predicted[b], all_observed ? 1U : observed);
    for (unsigned int cls = 0; cls < 3; ++cls) {
      float score = output.getVariableRef().getValue(b, 0, 0, cls);
      if (all_observed)
        EXPECT_FLOAT_EQ(score, cls == 1 ? 0.0f : -5.0f);
      else if (cls == observed)
        EXPECT_FLOAT_EQ(score, exact_match ? 0.0f : -5.0f);
      else
        EXPECT_LT(score, output.getVariableRef().getValue(b, 0, 0, observed));
    }
  }
  EXPECT_EQ(saved_centroids, centroids.getVariableRef());
  EXPECT_EQ(saved_counts, counts.getVariableRef());
}

TEST_P(CentroidKNNTest, unobservedClassCannotBeatExactMatch_n) {
  for (unsigned int observed = 0; observed < 3; ++observed)
    checkClassification(GetParam(), observed, true, false);
}

TEST_P(CentroidKNNTest, unobservedClassCannotBeatNegativeDistance_n) {
  for (unsigned int observed = 0; observed < 3; ++observed)
    checkClassification(GetParam(), observed, false, false);
}

TEST_P(CentroidKNNTest, allObservedScores_p) {
  checkClassification(GetParam(), 0, false, true);
}

INSTANTIATE_TEST_SUITE_P(BatchSizes, CentroidKNNTest,
                         ::testing::Values(1U, 2U));

} // namespace
