// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_kv_prefix_path.cpp
 * @brief   The kv_prefix dump path is a contract with the app that computes it
 *          independently; pin it to values produced by the reference
 *          implementation of run_with_kv_prefix.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "kv_prefix_path.h"

#include <gtest/gtest.h>

#include <string>

using namespace quick_dot_ai;

// Expected names come from compiling the reference loop verbatim (offset basis
// 1469598103934665603, prime 1099511628211) over these inputs.
TEST(KvPrefixPathTest, FileNameMatchesReferenceImplementation) {
  EXPECT_EQ(kv_prefix_file_name(""), "14650fb0739d0383.kv");
  EXPECT_EQ(kv_prefix_file_name("a"), "44bd8ad473cd9906.kv");
  EXPECT_EQ(kv_prefix_file_name("hello"), "005a0d15131ec7a1.kv");
  EXPECT_EQ(
    kv_prefix_file_name("<bos><|turn>system\nYou are an agent.<turn|>\n"),
    "a8054883ed660f69.kv");
  EXPECT_EQ(kv_prefix_file_name("한국어 정적 프롬프트\n"),
            "aa1b81f518530044.kv");
  EXPECT_EQ(kv_prefix_file_name("\xff\x80 bytes"), "67d123f50bd11d97.kv");
}

TEST(KvPrefixPathTest, HashIsOverBytesNotCharSign) {
  // bytes >= 0x80 must be hashed as unsigned values
  EXPECT_EQ(kv_prefix_hash("\xff"),
            (1469598103934665603ULL ^ 0xffULL) * 1099511628211ULL);
}

TEST(KvPrefixPathTest, PathIsModelDirKvPrefixName) {
  EXPECT_EQ(kv_prefix_path("/m/gemma4-26b-a4b-htp-w4a32", "hello"),
            "/m/gemma4-26b-a4b-htp-w4a32/kv_prefix/005a0d15131ec7a1.kv");
  EXPECT_EQ(kv_prefix_path("/m/pack/", "hello"),
            "/m/pack/kv_prefix/005a0d15131ec7a1.kv");
}

TEST(KvPrefixPathTest, TokensFromDescribeFile) {
  EXPECT_EQ(
    kv_describe_tokens("'/p/x.kv' v3 model='g' tokens=1537 layers=30 batch=1 "
                       "capacity=2304 dtype=1 geometry=77"),
    1537);
  EXPECT_EQ(kv_describe_tokens("'x' v1 model='(unnamed)' tokens=0 layers=1"),
            0);
  EXPECT_EQ(kv_describe_tokens("cannot open '/p/x.kv'"), -1);
  EXPECT_EQ(kv_describe_tokens("'/p/x.kv' has no KV-cache header (legacy "
                               "headerless file: no model, geometry or token "
                               "length recorded)"),
            -1);
  EXPECT_EQ(kv_describe_tokens("x tokens=abc"), -1);
  EXPECT_EQ(kv_describe_tokens("x tokens=99999999999999999999"), -1);
}
