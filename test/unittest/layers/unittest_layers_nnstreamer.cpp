// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_nnstreamer.cpp
 * @date 12 June 2021
 * @brief NNStreamer Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <dlfcn.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layers_common_tests.h>
#include <nnstreamer_layer.h>
#include <nntrainer_test_util.h>
#include <var_grad.h>

// auto semantic_nnstreamer = LayerSemanticsParamType(
//   nntrainer::createLayer<nntrainer::NNStreamerLayer>,
//   nntrainer::NNStreamerLayer::type,
//   {"model_path=../test/test_models/models/add.tflite"},
//   LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

// GTEST_PARAMETER_TEST(NNStreamer, LayerSemantics,
//                      ::testing::Values(semantic_nnstreamer));

/**
 * The ml_single_* functions below replace the ml-api single-shot API for this
 * test binary (the executable's definitions take precedence over the ones in
 * libcapi-ml-inference), so NNStreamerLayer can be driven without an
 * nnstreamer sub-plugin. ml_tensors_data_destroy() is wrapped to track the
 * output handles handed out by ml_single_invoke(), and
 * ml_tensors_data_get_tensor_data() to make reading them fail on demand.
 * The wrappers reach the real functions with dlsym(RTLD_NEXT), so ml-api must
 * be linked as shared libraries; otherwise the tests fail instead of passing.
 * These replacements are process-wide: any single-shot API use anywhere in
 * the unittest_layers binary gets this mock, not a real model.
 */
namespace {

constexpr unsigned int kWidth = 4;
int fake_single;
unsigned int invoke_count = 0;
int invoke_status = ML_ERROR_NONE;
unsigned int invoke_out_width = kWidth;
int get_data_status = ML_ERROR_NONE;
std::set<ml_tensors_data_h> live_outputs;
std::set<ml_tensors_data_h> freed_outputs;
unsigned int double_destroy_count = 0;

int createInfo(ml_tensors_info_h *info, unsigned int width) {
  ml_tensor_dimension dim;
  for (unsigned int i = 0; i < ML_TENSOR_RANK_LIMIT; ++i)
    dim[i] = 1;
  dim[0] = width;

  int status = ml_tensors_info_create(info);
  if (status != ML_ERROR_NONE)
    return status;
  if ((status = ml_tensors_info_set_count(*info, 1)) != ML_ERROR_NONE ||
      (status = ml_tensors_info_set_tensor_type(
         *info, 0, ML_TENSOR_TYPE_FLOAT32)) != ML_ERROR_NONE ||
      (status = ml_tensors_info_set_tensor_dimension(*info, 0, dim)) !=
        ML_ERROR_NONE)
    ml_tensors_info_destroy(*info);
  return status;
}

void resetMock() {
  invoke_count = 0;
  invoke_status = ML_ERROR_NONE;
  invoke_out_width = kWidth;
  get_data_status = ML_ERROR_NONE;
  live_outputs.clear();
  freed_outputs.clear();
  double_destroy_count = 0;
}

} // namespace

extern "C" {

int ml_single_open(ml_single_h *single, const char *model,
                   const ml_tensors_info_h input_info,
                   const ml_tensors_info_h output_info, ml_nnfw_type_e nnfw,
                   ml_nnfw_hw_e hw) {
  *single = &fake_single;
  return ML_ERROR_NONE;
}

int ml_single_close(ml_single_h single) {
  return single == &fake_single ? ML_ERROR_NONE : ML_ERROR_INVALID_PARAMETER;
}

int ml_single_get_input_info(ml_single_h single, ml_tensors_info_h *info) {
  return createInfo(info, kWidth);
}

int ml_single_get_output_info(ml_single_h single, ml_tensors_info_h *info) {
  return createInfo(info, kWidth);
}

int ml_single_invoke(ml_single_h single, const ml_tensors_data_h input,
                     ml_tensors_data_h *output) {
  /* like ml-api, the busy/closed-handle errors return before *output is set */
  if (invoke_status != ML_ERROR_NONE)
    return invoke_status;
  *output = nullptr;

  void *in_raw;
  size_t in_size;
  int status = ml_tensors_data_get_tensor_data(input, 0, &in_raw, &in_size);
  if (status != ML_ERROR_NONE)
    return status;

  ml_tensors_info_h out_info;
  status = createInfo(&out_info, invoke_out_width);
  if (status != ML_ERROR_NONE)
    return status;
  ml_tensors_data_h out_data;
  status = ml_tensors_data_create(out_info, &out_data);
  ml_tensors_info_destroy(out_info);
  if (status != ML_ERROR_NONE)
    return status;

  void *out_raw;
  size_t out_size;
  status = ml_tensors_data_get_tensor_data(out_data, 0, &out_raw, &out_size);
  if (status != ML_ERROR_NONE) {
    ml_tensors_data_destroy(out_data);
    return status;
  }
  float *in = static_cast<float *>(in_raw);
  float *out = static_cast<float *>(out_raw);
  for (size_t i = 0; i < out_size / sizeof(float); ++i)
    out[i] = (i < in_size / sizeof(float) ? in[i] : 0.0f) + 1.0f;

  ++invoke_count;
  live_outputs.insert(out_data);
  freed_outputs.erase(out_data);
  *output = out_data;
  return ML_ERROR_NONE;
}

int ml_tensors_data_destroy(ml_tensors_data_h data) {
  using destroy_fn = int (*)(ml_tensors_data_h);
  static destroy_fn real_destroy =
    reinterpret_cast<destroy_fn>(dlsym(RTLD_NEXT, "ml_tensors_data_destroy"));

  if (!real_destroy) {
    ADD_FAILURE() << "ml_tensors_data_destroy() of ml-api is not found";
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (freed_outputs.count(data)) {
    ++double_destroy_count;
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (live_outputs.erase(data))
    freed_outputs.insert(data);

  return real_destroy(data);
}

int ml_tensors_data_get_tensor_data(ml_tensors_data_h data, unsigned int index,
                                    void **raw_data, size_t *data_size) {
  using get_fn = int (*)(ml_tensors_data_h, unsigned int, void **, size_t *);
  static get_fn real_get = reinterpret_cast<get_fn>(
    dlsym(RTLD_NEXT, "ml_tensors_data_get_tensor_data"));

  if (!real_get) {
    ADD_FAILURE() << "ml_tensors_data_get_tensor_data() of ml-api is not found";
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (get_data_status != ML_ERROR_NONE && live_outputs.count(data))
    return get_data_status;

  return real_get(data, index, raw_data, data_size);
}

} /* extern "C" */

/**
 * @brief Fixture that finalizes an NNStreamerLayer against the mocked
 * single-shot API and prepares a run context for forwarding
 */
class NNStreamerLayerHandle : public ::testing::Test {
protected:
  /**
   * @brief SetUp test cases here
   */
  void SetUp() override {
    resetMock();
    dim = nntrainer::TensorDim(1, 1, 1, kWidth);
    layer = std::make_unique<nntrainer::NNStreamerLayer>();
    layer->setProperty(
      {"model_path=" +
       getResPath("add.tflite", {"test", "test_models", "models"})});

    nntrainer::InitLayerContext init_context({dim}, {true}, false, "nns");
    ASSERT_NO_THROW(layer->finalize(init_context));

    in.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "in");
    out.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "out");
    run_context = std::make_unique<nntrainer::RunLayerContext>(
      "nns", false, 0.0f, false, 1.0f, nullptr, false,
      std::vector<nntrainer::Weight *>{},
      std::vector<nntrainer::Var_Grad *>{&in[0]},
      std::vector<nntrainer::Var_Grad *>{&out[0]},
      std::vector<nntrainer::Var_Grad *>{});
  }

  nntrainer::TensorDim dim;
  std::unique_ptr<nntrainer::NNStreamerLayer> layer;
  std::vector<nntrainer::Var_Grad> in;
  std::vector<nntrainer::Var_Grad> out;
  std::unique_ptr<nntrainer::RunLayerContext> run_context;
};

/**
 * @brief every forwarding releases the output handle ml_single_invoke()
 * allocated for it
 */
TEST_F(NNStreamerLayerHandle, forwarding_releases_output_handle) {
  nntrainer::Tensor &input = run_context->getInput(0);
  nntrainer::Tensor &output = run_context->getOutput(0);

  for (unsigned int iter = 1; iter <= 10; ++iter) {
    for (unsigned int i = 0; i < kWidth; ++i)
      input.getValue(i) = static_cast<float>(iter * 10 + i);

    ASSERT_NO_THROW(layer->forwarding(*run_context, false));
    EXPECT_EQ(invoke_count, iter);
    EXPECT_TRUE(live_outputs.empty());
    for (unsigned int i = 0; i < kWidth; ++i)
      EXPECT_FLOAT_EQ(output.getValue(i),
                      static_cast<float>(iter * 10 + i) + 1.0f);
  }

  layer.reset();
  EXPECT_TRUE(live_outputs.empty());
  EXPECT_EQ(double_destroy_count, 0u);
}

/**
 * @brief destroying the layer after forwarding does not destroy the released
 * output handle again
 */
TEST_F(NNStreamerLayerHandle, release_after_forwarding_no_double_destroy) {
  ASSERT_NO_THROW(layer->forwarding(*run_context, false));
  EXPECT_EQ(freed_outputs.size(), 1u);

  layer.reset();
  EXPECT_EQ(double_destroy_count, 0u);
  EXPECT_TRUE(live_outputs.empty());
}

/**
 * @brief output size mismatch throws and still releases the output handle
 */
TEST_F(NNStreamerLayerHandle, forwarding_output_size_mismatch_n) {
  invoke_out_width = kWidth * 2;
  EXPECT_THROW(layer->forwarding(*run_context, false), std::runtime_error);
  EXPECT_EQ(invoke_count, 1u);
  EXPECT_TRUE(live_outputs.empty());

  invoke_out_width = kWidth;
  EXPECT_NO_THROW(layer->forwarding(*run_context, false));
  EXPECT_TRUE(live_outputs.empty());

  layer.reset();
  EXPECT_EQ(double_destroy_count, 0u);
}

/**
 * @brief failing to read the invoke output throws and still releases the
 * output handle
 */
TEST_F(NNStreamerLayerHandle, forwarding_get_output_data_fail_n) {
  ASSERT_NO_THROW(layer->forwarding(*run_context, false));

  get_data_status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_THROW(layer->forwarding(*run_context, false), std::runtime_error);
  EXPECT_EQ(invoke_count, 2u);
  EXPECT_TRUE(live_outputs.empty());

  get_data_status = ML_ERROR_NONE;
  EXPECT_NO_THROW(layer->forwarding(*run_context, false));
  EXPECT_TRUE(live_outputs.empty());

  layer.reset();
  EXPECT_EQ(double_destroy_count, 0u);
}

/**
 * @brief failing invoke throws and leaves no output handle behind
 */
TEST_F(NNStreamerLayerHandle, forwarding_invoke_fail_n) {
  ASSERT_NO_THROW(layer->forwarding(*run_context, false));

  invoke_status = ML_ERROR_STREAMS_PIPE;
  EXPECT_THROW(layer->forwarding(*run_context, false), std::runtime_error);
  EXPECT_EQ(invoke_count, 1u);
  EXPECT_TRUE(live_outputs.empty());

  layer.reset();
  EXPECT_EQ(double_destroy_count, 0u);
}
