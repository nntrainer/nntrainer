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
 * @brief Test supportBackwarding returns false for EmbeddingLayer
 * @note EmbeddingLayer does not support backwarding
 */
TEST(EmbeddingLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_FALSE(layer->supportBackwarding());
}

/**
 * @brief Test setProperty and verify with getProperty
 * @details Verify that the value is properly set after setProperty by using
 * getProperty to check
 */
TEST(EmbeddingLayerDirect, setProperty_getProperty) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"in_dim=5000", "out_dim=128"}));

  // Verify properties are properly set using getProperty
  EXPECT_EQ(layer->getProperty("in_dim"), "5000");
  EXPECT_EQ(layer->getProperty("out_dim"), "128");
}

/**
 * @brief Test setProperty can be called multiple times (resetting)
 * @details After calling setProperty again, getProperty should return the new
 * values
 */
TEST(EmbeddingLayerDirect, setProperty_reset) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});

  // Set initial values
  EXPECT_NO_THROW(layer->setProperty({"in_dim=1000", "out_dim=64"}));

  // Verify initial values
  EXPECT_EQ(layer->getProperty("in_dim"), "1000");
  EXPECT_EQ(layer->getProperty("out_dim"), "64");

  // Reset with new values
  EXPECT_NO_THROW(layer->setProperty({"in_dim=5000", "out_dim=256"}));

  // Verify new values are properly set
  EXPECT_EQ(layer->getProperty("in_dim"), "5000");
  EXPECT_EQ(layer->getProperty("out_dim"), "256");
}

/**
 * @brief Test setProperty with invalid property name
 */
TEST(EmbeddingLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(EmbeddingLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  // in_dim expects a positive integer, passing a string
  EXPECT_THROW(layer->setProperty({"in_dim=not_a_number"}), std::exception);
}

/**
 * @brief Test with various valid dimensions
 */
TEST(EmbeddingLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"in_dim=5000", "out_dim=128"}));

  auto layer2 = nntrainer::createLayer<nntrainer::EmbeddingLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"in_dim=32000", "out_dim=256"}));
}
