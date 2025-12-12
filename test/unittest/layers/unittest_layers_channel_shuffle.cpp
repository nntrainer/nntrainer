// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_channel_shuffle.cpp
 * @date 12 December 2025
 * @brief Channel Shuffle Layer Test
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <channel_shuffle.h>
#include <layers_common_tests.h>

// Note: ChannelShuffle LayerSemantics tests are skipped because they require
// channels > split_number, but the default input has only 1 channel.


/**
 * @brief Direct test for ChannelShuffle setProperty
 */
TEST(ChannelShuffleDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_NO_THROW(layer->setProperty({"split_number=4"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(ChannelShuffleDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_EQ(layer->getType(), "channel_shuffle");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(ChannelShuffleDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different group values
 */
TEST(ChannelShuffleDirect, setProperty_various_groups) {
  auto layer1 = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_NO_THROW(layer1->setProperty({"split_number=2"}));

  auto layer2 = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_NO_THROW(layer2->setProperty({"split_number=8"}));

  auto layer3 = nntrainer::createLayer<nntrainer::ChannelShuffle>({});
  EXPECT_NO_THROW(layer3->setProperty({"split_number=16"}));
}
