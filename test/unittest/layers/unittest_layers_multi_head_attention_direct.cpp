// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_multi_head_attention_direct.cpp
 * @date 12 December 2025
 * @brief Multi Head Attention Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <multi_head_attention_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for MultiHeadAttentionLayer setProperty
 */
TEST(MultiHeadAttentionLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"num_heads=4", "projected_key_dim=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(MultiHeadAttentionLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>({});
  EXPECT_EQ(layer->getType(), "multi_head_attention");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(MultiHeadAttentionLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various head counts
 */
TEST(MultiHeadAttentionLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"num_heads=8", "projected_key_dim=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"num_heads=16", "projected_key_dim=128"}));
}
