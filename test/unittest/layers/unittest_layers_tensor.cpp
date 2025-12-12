// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_tensor.cpp
 * @date 12 December 2025
 * @brief Tensor Layer Test
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <tensor_layer.h>
#include <layers_common_tests.h>

// TensorLayer is a special layer for QNN and is not compatible with
// LayerSemantics framework, so we only test direct APIs.



/**
 * @brief Test setProperty with invalid property
 */
TEST(TensorLayerDirect, setProperty_invalid) {
  auto layer = nntrainer::createLayer<nntrainer::TensorLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_prop=value"}),
               std::invalid_argument);
}

/**
 * @brief Test getType returns correct type
 */
TEST(TensorLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::TensorLayer>({});
  EXPECT_EQ(layer->getType(), "tensor");
}

/**
 * @brief Test supportBackwarding returns false
 */
TEST(TensorLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::TensorLayer>({});
  EXPECT_FALSE(layer->supportBackwarding());
}
