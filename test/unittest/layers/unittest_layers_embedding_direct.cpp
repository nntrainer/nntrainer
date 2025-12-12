// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_embedding_direct.cpp
 * @date 12 December 2025
 * @brief Embedding Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <embedding.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for EmbeddingLayer setProperty
 */
TEST(EmbeddingLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"in_dim=1000", "out_dim=64"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(EmbeddingLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_EQ(layer->getType(), "embedding");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(EmbeddingLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different dimensions
 */
TEST(EmbeddingLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"in_dim=5000", "out_dim=128"}));

  auto layer2 = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"in_dim=32000", "out_dim=256"}));
}
