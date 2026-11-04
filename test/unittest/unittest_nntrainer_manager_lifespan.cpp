// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_nntrainer_manager_lifespan.cpp
 * @date   28 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs
 * @brief  Tests for the two private rules Manager::requestInputs() uses to
 *         decide how long a node's inputs and their gradients live:
 *         resolveInputLifespans() (the rule itself) and inputLifespansFor()
 *         (the wiring that reads a real LayerNode and calls the rule).
 *
 * @details ManagerInputLifespanTest is declared a friend of Manager (see
 * manager.h) so these private methods stay reachable from a test without
 * widening Manager's public API.
 */

#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <layer_node.h>
#include <manager.h>

namespace nntrainer {

using LS = TensorLifespan;

/**
 * @brief friend fixture giving the tests access to Manager's private
 * resolveInputLifespans() and inputLifespansFor()
 */
class ManagerInputLifespanTest : public ::testing::Test {
protected:
  /**
   * @brief forwards to the private Manager::resolveInputLifespans()
   */
  static std::pair<LS, LS> resolve(bool is_train, bool trainable,
                                   std::optional<LS> var_override,
                                   std::optional<LS> grad_override) {
    return Manager::resolveInputLifespans(is_train, trainable, var_override,
                                          grad_override);
  }

  /**
   * @brief forwards to the private Manager::inputLifespansFor()
   */
  static std::pair<LS, LS> inputLifespansFor(const LayerNode *consumer_lnode,
                                             bool is_train, bool trainable) {
    return Manager::inputLifespansFor(consumer_lnode, is_train, trainable);
  }
};

namespace {

/**
 * @brief which hardcoded type list a layer fell into before the change
 */
enum class Group {
  NONE,         /**< no list */
  FORWARD_ONLY, /**< activation, multiout, batch_normalization,
                   layer_normalization */
  LOSS,         /**< mse, cross_softmax, cross_sigmoid */
};

/**
 * @brief the decision Manager::requestInputs() made before the change
 */
std::pair<LS, LS> legacyLifespans(bool is_train, bool trainable, Group group,
                                  bool is_grucell) {
  LS var = is_train ? LS::FORWARD_GRAD_LIFESPAN : LS::FORWARD_FUNC_LIFESPAN;
  LS grad = LS::CALC_DERIV_LIFESPAN;
  if (group == Group::FORWARD_ONLY || !trainable)
    var = LS::FORWARD_FUNC_LIFESPAN;
  if (group == Group::LOSS)
    var = LS::FORWARD_DERIV_LIFESPAN;
  if (is_grucell)
    grad = LS::CALC_GRAD_DERIV_LIFESPAN;
  return {var, grad};
}

/**
 * @brief what a layer of that group declares after the change
 */
std::optional<LS> declaredVar(Group group) {
  switch (group) {
  case Group::FORWARD_ONLY:
    return LS::FORWARD_FUNC_LIFESPAN;
  case Group::LOSS:
    return LS::FORWARD_DERIV_LIFESPAN;
  default:
    return std::nullopt;
  }
}

} // namespace

/**
 * @brief every mode, trainability and declaration resolves as it did before
 */
TEST_F(ManagerInputLifespanTest, resolveInputLifespans_matches_legacy_p) {
  int checked = 0;
  for (bool is_train : {false, true}) {
    for (bool trainable : {false, true}) {
      for (Group group : {Group::NONE, Group::FORWARD_ONLY, Group::LOSS}) {
        for (bool is_grucell : {false, true}) {
          SCOPED_TRACE("is_train=" + std::to_string(is_train) +
                       " trainable=" + std::to_string(trainable) +
                       " group=" + std::to_string(static_cast<int>(group)) +
                       " grucell=" + std::to_string(is_grucell));
          std::optional<LS> grad_decl =
            is_grucell ? std::optional<LS>(LS::CALC_GRAD_DERIV_LIFESPAN)
                       : std::nullopt;
          auto expected =
            legacyLifespans(is_train, trainable, group, is_grucell);
          auto actual =
            resolve(is_train, trainable, declaredVar(group), grad_decl);
          EXPECT_EQ(actual.first, expected.first);
          EXPECT_EQ(actual.second, expected.second);
          ++checked;
        }
      }
    }
  }
  EXPECT_EQ(checked, 24);
}

/**
 * @brief inputLifespansFor() reads a real LayerNode's declarations and
 * passes them to resolveInputLifespans() in the right slots -- this would
 * fail if the var/grad arguments were swapped, or the wrong node read
 */
TEST_F(ManagerInputLifespanTest, inputLifespansFor_reads_real_layers_p) {
  // activation: declares only getInputLifespan() -> FORWARD_FUNC
  auto activation = createLayerNode("activation");
  EXPECT_EQ(inputLifespansFor(activation.get(), /*is_train=*/true,
                              /*trainable=*/true),
            std::make_pair(LS::FORWARD_FUNC_LIFESPAN, LS::CALC_DERIV_LIFESPAN));

  // mse: declares only getInputLifespan() -> FORWARD_DERIV, and the
  // declaration wins even though the node is trainable and training
  auto mse = createLayerNode("mse");
  EXPECT_EQ(
    inputLifespansFor(mse.get(), /*is_train=*/true, /*trainable=*/true),
    std::make_pair(LS::FORWARD_DERIV_LIFESPAN, LS::CALC_DERIV_LIFESPAN));

  // grucell: declares only getInputGradLifespan() -> CALC_GRAD_DERIV; the
  // variable lifespan falls through to the is_train default, proving the
  // var/grad declarations are not swapped. No properties are needed to
  // construct the node (see unittest_layer_node.cpp's byType("grucell")).
  auto grucell = createLayerNode("grucell");
  EXPECT_EQ(
    inputLifespansFor(grucell.get(), /*is_train=*/true, /*trainable=*/true),
    std::make_pair(LS::FORWARD_GRAD_LIFESPAN, LS::CALC_GRAD_DERIV_LIFESPAN));

  // a GraphNode that is not a LayerNode (consumer_lnode == nullptr) always
  // gets the mode/trainable default, with no declaration applied
  EXPECT_EQ(inputLifespansFor(nullptr, /*is_train=*/true, /*trainable=*/true),
            std::make_pair(LS::FORWARD_GRAD_LIFESPAN, LS::CALC_DERIV_LIFESPAN));
}

} // namespace nntrainer

/**
 * @brief Main gtest
 */
GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
