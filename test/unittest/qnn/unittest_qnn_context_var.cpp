// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_qnn_context_var.cpp
 * @date    30 Sep 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 * @brief   Host unit test of QNNVar::makeContext() in
 * nntrainer/qnn/jni/qnn_context_var.h. The header is compiled against the
 * stub SDK declarations in ./stub and driven through fake QNN function
 * pointers, so no Qualcomm SDK or HTP device is needed. malloc, calloc and
 * free are wrapped at link time (-Wl,--wrap) so that the tests can check
 * which heap blocks makeContext() leaves allocated.
 */

#include <gtest/gtest.h>

#include <fcntl.h>
#include <string>
#include <unistd.h>

#include "qnn_context_var.h"

using nntrainer::QNNVar;
using nntrainer::StatusCode;
using qnn_wrapper_api::GraphInfo_t;

namespace {

constexpr size_t kMaxBlocks = 1024;
void *g_blocks[kMaxBlocks];
bool g_tracking = false;
size_t g_untracked = 0;

/** @brief remember a block allocated while tracking is on */
void addBlock(void *p) {
  if (!g_tracking || p == nullptr)
    return;
  for (auto &b : g_blocks) {
    if (b == nullptr) {
      b = p;
      return;
    }
  }
  g_untracked++;
}

/** @brief forget a block that is being freed */
void removeBlock(void *p) {
  for (auto &b : g_blocks) {
    if (b == p) {
      b = nullptr;
      return;
    }
  }
}

/** @brief whether a tracked block is still allocated */
bool isLive(void *p) {
  if (p == nullptr)
    return false;
  for (auto b : g_blocks)
    if (b == p)
      return true;
  return false;
}

/** @brief number of tracked blocks still allocated */
size_t liveBlocks() {
  size_t n = 0;
  for (auto b : g_blocks)
    if (b != nullptr)
      n++;
  return n + g_untracked;
}

} // namespace

extern "C" {
void *__real_malloc(size_t size);
void *__real_calloc(size_t n, size_t size);
void __real_free(void *p);

/** @brief link-time wrapper of malloc */
void *__wrap_malloc(size_t size) {
  void *p = __real_malloc(size);
  addBlock(p);
  return p;
}

/** @brief link-time wrapper of calloc */
void *__wrap_calloc(size_t n, size_t size) {
  void *p = __real_calloc(n, size);
  addBlock(p);
  return p;
}

/** @brief link-time wrapper of free */
void __wrap_free(void *p) {
  removeBlock(p);
  __real_free(p);
}
}

namespace {

/** @brief knobs and counters of the fake QNN functions */
struct Fake {
  bool failSystemContextCreate = false;
  bool failGetBinaryInfo = false;
  bool failCopyMetadata = false;
  bool failCreateFromBinary = false;
  bool createWritesHandle = false;
  bool failContextFree = false;
  uint32_t numGraphs = 2;
  int contextCreateCalls = 0;
  int contextFreeCalls = 0;
  int beforeCalls = 0;
  int afterCalls = 0;
};
Fake g_fake;

const QnnSystemContext_BinaryInfo_t g_binaryInfo{1};

Qnn_ErrorHandle_t fakeSystemContextCreate(QnnSystemContext_Handle_t *h) {
  if (g_fake.failSystemContextCreate)
    return 1;
  *h = malloc(8);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t
fakeGetBinaryInfo(QnnSystemContext_Handle_t, void *, uint64_t,
                  const QnnSystemContext_BinaryInfo_t **binaryInfo,
                  Qnn_ContextBinarySize_t *size) {
  if (g_fake.failGetBinaryInfo)
    return 1;
  *binaryInfo = &g_binaryInfo;
  *size = sizeof(g_binaryInfo);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t fakeSystemContextFree(QnnSystemContext_Handle_t h) {
  free(h);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t fakeCreateFromBinary(Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                       const QnnContext_Config_t **,
                                       const void *, Qnn_ContextBinarySize_t,
                                       Qnn_ContextHandle_t *context,
                                       Qnn_ProfileHandle_t) {
  g_fake.contextCreateCalls++;
  if (!g_fake.failCreateFromBinary || g_fake.createWritesHandle)
    *context = malloc(8);
  return g_fake.failCreateFromBinary ? 1 : QNN_SUCCESS;
}

Qnn_ErrorHandle_t fakeContextFree(Qnn_ContextHandle_t context,
                                  Qnn_ProfileHandle_t) {
  g_fake.contextFreeCalls++;
  free(context);
  return g_fake.failContextFree ? 1 : QNN_CONTEXT_NO_ERROR;
}

/** @brief backend extension that hands out configs it keeps owning */
class FakeExtension : public QnnBackendExtensionStub {
public:
  /** @brief allocate the extension-owned configs */
  FakeExtension() {
    configs = (QnnContext_Config_t **)malloc(2 * sizeof(QnnContext_Config_t *));
    configs[0] = (QnnContext_Config_t *)malloc(sizeof(QnnContext_Config_t));
    configs[1] = (QnnContext_Config_t *)malloc(sizeof(QnnContext_Config_t));
  }
  /** @brief release the extension-owned configs */
  ~FakeExtension() {
    free(configs[0]);
    free(configs[1]);
    free(configs);
  }
  /** @copydoc QnnBackendExtensionStub::beforeCreateFromBinary */
  bool beforeCreateFromBinary(QnnContext_Config_t ***customConfigs,
                              uint32_t *customConfigCount) override {
    g_fake.beforeCalls++;
    *customConfigs = configs;
    *customConfigCount = 2;
    return true;
  }
  /** @copydoc QnnBackendExtensionStub::afterCreateFromBinary */
  bool afterCreateFromBinary() override {
    g_fake.afterCalls++;
    return true;
  }
  QnnContext_Config_t **configs; /**< owned by this extension */
};

} // namespace

bool qnn_wrapper_api::freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t) {
  free(tensors);
  tensors = nullptr;
  return true;
}

/**
 * Allocates the graph tables the way the QNN SDK sample does: an array of
 * pointers into one array of GraphInfo_t, with a name and tensor arrays per
 * graph. On failure it leaves the output untouched.
 */
bool qnn::tools::sample_app::copyMetadataToGraphsInfo(
  const QnnSystemContext_BinaryInfo_t *, GraphInfo_t **&graphsInfo,
  uint32_t &graphsCount) {
  graphsCount = 0;
  if (g_fake.failCopyMetadata)
    return false;
  uint32_t n = g_fake.numGraphs;
  graphsInfo = (GraphInfo_t **)calloc(n, sizeof(GraphInfo_t *));
  if (n == 0)
    return true;
  auto *graphs = (GraphInfo_t *)calloc(n, sizeof(GraphInfo_t));
  for (uint32_t i = 0; i < n; i++) {
    std::string name = "graph_" + std::to_string(i);
    graphs[i].graphName = (char *)malloc(name.size() + 1);
    memcpy(graphs[i].graphName, name.c_str(), name.size() + 1);
    graphs[i].inputTensors = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    graphs[i].numInputTensors = 1;
    graphs[i].outputTensors = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    graphs[i].numOutputTensors = 1;
    graphsInfo[i] = graphs + i;
  }
  graphsCount = n;
  return true;
}

/** @brief fixture that wires QNNVar to the fakes and a real binary file */
class QnnMakeContext : public ::testing::Test {
protected:
  /** @brief create the context binary file and reset the fakes */
  void SetUp() override {
    std::string path = testing::TempDir() + "unittest_qnn_context_var_XXXXXX";
    int fd = mkstemp(&path[0]);
    ASSERT_GE(fd, 0) << "cannot create " << path;
    std::string data(4096, 'q');
    ASSERT_EQ(write(fd, data.data(), data.size()), (ssize_t)data.size());
    close(fd);
    bin = path;

    g_fake = Fake();
    auto &sys = var.m_qnnFunctionPointers.qnnSystemInterface;
    sys.systemContextCreate = fakeSystemContextCreate;
    sys.systemContextGetBinaryInfo = fakeGetBinaryInfo;
    sys.systemContextFree = fakeSystemContextFree;
    auto &qnn = var.m_qnnFunctionPointers.qnnInterface;
    qnn.contextCreateFromBinary = fakeCreateFromBinary;
    qnn.contextFree = fakeContextFree;
    var.m_profilingLevel = qnn::tools::sample_app::ProfilingLevel::OFF;

    for (auto &b : g_blocks)
      b = nullptr;
    g_untracked = 0;
    g_tracking = true;
  }

  /** @brief stop tracking and remove the binary file */
  void TearDown() override {
    g_tracking = false;
    unlink(bin.c_str());
  }

  /** @brief expect a failed makeContext() to leave nothing behind */
  void expectCleanFailure(StatusCode status) {
    EXPECT_EQ(status, StatusCode::FAILURE);
    EXPECT_FALSE(var.findContext(bin).has_value());
    EXPECT_TRUE(var.ct_map.empty());
    EXPECT_EQ(liveBlocks(), 0u);
  }

  QNNVar var;      /**< object under test */
  std::string bin; /**< context binary path */
};

TEST_F(QnnMakeContext, cachesContextAndFreesIt) {
  EXPECT_EQ(var.makeContext(bin), StatusCode::SUCCESS);
  auto ctx = var.findContext(bin);
  ASSERT_TRUE(ctx.has_value());
  EXPECT_NE(ctx->get().m_context, nullptr);
  EXPECT_EQ(ctx->get().m_graphsCount, 2u);
  EXPECT_NE(ctx->get().getGraphPtr("graph_0"), nullptr);
  EXPECT_NE(ctx->get().getGraphPtr("graph_1"), nullptr);
  EXPECT_GT(liveBlocks(), 0u);

  EXPECT_EQ(var.freeContext(bin), StatusCode::SUCCESS);
  EXPECT_EQ(g_fake.contextFreeCalls, 1);
  EXPECT_EQ(liveBlocks(), 0u);
}

TEST_F(QnnMakeContext, cachesContextWithoutGraphsAndFreesIt) {
  g_fake.numGraphs = 0;
  EXPECT_EQ(var.makeContext(bin), StatusCode::SUCCESS);
  auto ctx = var.findContext(bin);
  ASSERT_TRUE(ctx.has_value());
  EXPECT_EQ(ctx->get().m_graphsCount, 0u);
  EXPECT_TRUE(ctx->get().graph_map.empty());

  EXPECT_EQ(var.freeContext(bin), StatusCode::SUCCESS);
  EXPECT_EQ(g_fake.contextFreeCalls, 1);
  EXPECT_EQ(liveBlocks(), 0u);
}

TEST_F(QnnMakeContext, retrySucceedsAfterFailure) {
  g_fake.failCreateFromBinary = true;
  expectCleanFailure(var.makeContext(bin));

  g_fake.failCreateFromBinary = false;
  EXPECT_EQ(var.makeContext(bin), StatusCode::SUCCESS);
  auto ctx = var.findContext(bin);
  ASSERT_TRUE(ctx.has_value());
  EXPECT_NE(ctx->get().m_context, nullptr);
  EXPECT_NE(ctx->get().getGraphPtr("graph_1"), nullptr);
  EXPECT_EQ(g_fake.contextCreateCalls, 2);

  EXPECT_EQ(var.freeContext(bin), StatusCode::SUCCESS);
  EXPECT_EQ(liveBlocks(), 0u);
}

TEST_F(QnnMakeContext, systemContextCreateFails_n) {
  g_fake.failSystemContextCreate = true;
  expectCleanFailure(var.makeContext(bin));
  EXPECT_EQ(g_fake.contextCreateCalls, 0);
}

TEST_F(QnnMakeContext, getBinaryInfoFails_n) {
  g_fake.failGetBinaryInfo = true;
  expectCleanFailure(var.makeContext(bin));
  EXPECT_EQ(g_fake.contextCreateCalls, 0);
}

TEST_F(QnnMakeContext, copyMetadataFails_n) {
  g_fake.failCopyMetadata = true;
  expectCleanFailure(var.makeContext(bin));
  EXPECT_EQ(g_fake.contextCreateCalls, 0);
}

TEST_F(QnnMakeContext, createFromBinaryMissing_n) {
  var.m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary = nullptr;
  expectCleanFailure(var.makeContext(bin));
}

TEST_F(QnnMakeContext, createFromBinaryFails_n) {
  g_fake.failCreateFromBinary = true;
  expectCleanFailure(var.makeContext(bin));
  EXPECT_EQ(g_fake.contextFreeCalls, 0);
}

TEST_F(QnnMakeContext, createFromBinaryFailsWithHandle_n) {
  g_fake.failCreateFromBinary = true;
  g_fake.createWritesHandle = true;
  expectCleanFailure(var.makeContext(bin));
  EXPECT_EQ(g_fake.contextFreeCalls, 1);
}

TEST_F(QnnMakeContext, createFromBinaryFailsAndContextFreeFails_n) {
  g_fake.failCreateFromBinary = true;
  g_fake.createWritesHandle = true;
  g_fake.failContextFree = true;
  StatusCode status = StatusCode::SUCCESS;
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(status = var.makeContext(bin));
  std::string log = testing::internal::GetCapturedStderr();
  expectCleanFailure(status);
  EXPECT_EQ(g_fake.contextFreeCalls, 1);
  EXPECT_NE(log.find("Failed to free QNN context"), std::string::npos);
}

TEST_F(QnnMakeContext, createFromBinaryFailsWithoutGraphs_n) {
  g_fake.failCreateFromBinary = true;
  g_fake.numGraphs = 0;
  expectCleanFailure(var.makeContext(bin));
}

TEST_F(QnnMakeContext, failureKeepsExtensionConfigs_n) {
  FakeExtension ext;
  BackendExtensions holder;
  holder.iface = &ext;
  var.m_backendExtensions = &holder;
  g_fake.failCreateFromBinary = true;

  EXPECT_EQ(var.makeContext(bin), StatusCode::FAILURE);
  EXPECT_TRUE(var.ct_map.empty());
  EXPECT_EQ(g_fake.beforeCalls, 1);
  EXPECT_EQ(g_fake.afterCalls, 1);
  EXPECT_TRUE(isLive(ext.configs));
  EXPECT_TRUE(isLive(ext.configs[0]));
  EXPECT_TRUE(isLive(ext.configs[1]));
  EXPECT_EQ(liveBlocks(), 3u);
}
