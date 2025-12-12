// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 UGyeong Song <thddnrud@snu.ac.kr>
 *
 * @file unittest_layers_convolution.cpp
 * @date 21 November 2024
 * @brief Conv2dTranspose Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author UGyeong Song <thddnrud@snu.ac.kr>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <conv2d_transpose_layer.h>
#include <layers_common_tests.h>

auto semantic_conv2d_transpose = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  nntrainer::Conv2DTransposeLayer::type,
  {"filters=1", "kernel_size=1,1", "padding=1,1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Convolution2DTranspose, LayerSemantics,
                     ::testing::Values(semantic_conv2d_transpose));

auto conv2d_transpose_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4",
  "conv2d_transpose_sb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4",
  "conv2d_transpose_mb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "conv2d_transpose_sb_same_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "conv2d_transpose_mb_same_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "conv2d_transpose_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "conv2d_transpose_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "conv2d_transpose_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "conv2d_transpose_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "1:3:7:7", "conv2d_transpose_sb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "3:3:7:7", "conv2d_transpose_mb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "conv2d_transpose_sb_no_overlap.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=3",
    "kernel_size=2,2",
    "stride=3,3",
  },
  "3:2:5:5", "conv2d_transpose_mb_no_overlap.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "conv2d_transpose_sb_1x1_kernel.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=3",
    "kernel_size=1,1",
    "stride=2,2",
  },
  "3:2:5:5", "conv2d_transpose_mb_1x1_kernel.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_transpose_sb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_transpose_mb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_sb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_transpose_sb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_transpose_mb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_transpose_mb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(
  Convolution2DTranspose, LayerGoldenTest,
  ::testing::Values(
    conv2d_transpose_sb_minimum, conv2d_transpose_mb_minimum,
    conv2d_transpose_sb_same_remain, conv2d_transpose_mb_same_remain,
    conv2d_transpose_sb_same_uneven_remain_1,
    conv2d_transpose_sb_same_uneven_remain_2,
    conv2d_transpose_mb_same_uneven_remain_1,
    conv2d_transpose_mb_same_uneven_remain_2,
    conv2d_transpose_sb_valid_drop_last, conv2d_transpose_mb_valid_drop_last,
    conv2d_transpose_sb_no_overlap, conv2d_transpose_mb_no_overlap,
    conv2d_transpose_sb_1x1_kernel, conv2d_transpose_mb_1x1_kernel,
    conv2d_transpose_sb_dilation, conv2d_transpose_mb_dilation,
    conv2d_transpose_sb_same_dilation, conv2d_transpose_mb_same_dilation));

#ifdef ENABLE_FP16
auto conv2d_transpose_sb_minimum_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4",
  "conv2d_transpose_sb_minimum_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_minimum_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4",
  "conv2d_transpose_mb_minimum_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_same_remain_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "conv2d_transpose_sb_same_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_same_remain_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "conv2d_transpose_mb_same_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_same_uneven_remain_1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "conv2d_transpose_sb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_same_uneven_remain_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "conv2d_transpose_sb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_same_uneven_remain_1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "conv2d_transpose_mb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_same_uneven_remain_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "conv2d_transpose_mb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_valid_drop_last_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "1:3:7:7", "conv2d_transpose_sb_valid_drop_last_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_valid_drop_last_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "3:3:7:7", "conv2d_transpose_mb_valid_drop_last_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_no_overlap_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "conv2d_transpose_sb_no_overlap_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_no_overlap_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=3",
    "kernel_size=2,2",
    "stride=3,3",
  },
  "3:2:5:5", "conv2d_transpose_mb_no_overlap_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_1x1_kernel_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "conv2d_transpose_sb_1x1_kernel_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_1x1_kernel_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=3",
    "kernel_size=1,1",
    "stride=2,2",
  },
  "3:2:5:5", "conv2d_transpose_mb_1x1_kernel_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_transpose_sb_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_transpose_mb_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_sb_same_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_transpose_sb_same_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_transpose_mb_same_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_transpose_mb_same_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(
  Convolution2DTranspose16, LayerGoldenTest,
  ::testing::Values(conv2d_transpose_sb_minimum_w16a16,
                    conv2d_transpose_mb_minimum_w16a16,
                    conv2d_transpose_sb_same_remain_w16a16,
                    conv2d_transpose_mb_same_remain_w16a16,
                    conv2d_transpose_sb_same_uneven_remain_1_w16a16,
                    conv2d_transpose_sb_same_uneven_remain_2_w16a16,
                    conv2d_transpose_mb_same_uneven_remain_1_w16a16,
                    conv2d_transpose_mb_same_uneven_remain_2_w16a16,
                    conv2d_transpose_sb_valid_drop_last_w16a16,
                    conv2d_transpose_mb_valid_drop_last_w16a16,
                    conv2d_transpose_sb_no_overlap_w16a16,
                    conv2d_transpose_mb_no_overlap_w16a16,
                    conv2d_transpose_sb_1x1_kernel_w16a16,
                    conv2d_transpose_mb_1x1_kernel_w16a16,
                    conv2d_transpose_sb_dilation_w16a16,
                    conv2d_transpose_mb_dilation_w16a16,
                    conv2d_transpose_sb_same_dilation_w16a16,
                    conv2d_transpose_mb_same_dilation_w16a16));
#endif

/**
 * @brief Direct test for Conv2DTransposeLayer setProperty
 */
TEST(Conv2DTransposeLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  EXPECT_NO_THROW(layer->setProperty(
    {"filters=16", "kernel_size=3,3", "stride=2,2", "padding=1,1"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(Conv2DTransposeLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  EXPECT_EQ(layer->getType(), "conv2dtranspose");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(Conv2DTransposeLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test various property combinations
 */
TEST(Conv2DTransposeLayerDirect, setProperty_various) {
  auto layer = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  
  // Test with dilation
  EXPECT_NO_THROW(layer->setProperty(
    {"filters=8", "kernel_size=5,5", "dilation=2,2"}));
  
  // Test with same padding
  auto layer2 = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  EXPECT_NO_THROW(layer2->setProperty(
    {"filters=4", "kernel_size=3,3", "padding=same"}));
  
  // Test with valid padding
  auto layer3 = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>({});
  EXPECT_NO_THROW(layer3->setProperty(
    {"filters=4", "kernel_size=3,3", "padding=valid"}));
}
