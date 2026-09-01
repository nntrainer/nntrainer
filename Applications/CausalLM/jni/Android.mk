LOCAL_PATH := $(call my-dir)
CAUSALLM_JNI_PATH := $(LOCAL_PATH)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

# Common Includes Definition
#
# The three $(NNTRAINER_ROOT) entries reach headers that libnntrainer keeps
# private and deliberately does not install, so the ndk build has to find them
# in the nntrainer source dir rather than through the prebuilt include export
# (the meson leg does not notice because it compiles against nntrainer_inc):
#   nntrainer/tensor              residency_policy.h, the application-declared
#                                 residency boundaries causal_lm.cpp populates
#                                 before the graph allocates
#   nntrainer/tensor/cl_operations  TEMPORARY: the layer TUs here (mha_core,
#                                 reshaped_rms_norm) still include the raw
#                                 OpenCL kernel wrappers <blas_kernels.h> /
#                                 <attention_kernels.h> instead of going
#                                 through the ComputeOps table. Delete that
#                                 entry together with the last raw
#                                 nntrainer::*_cl(...) call site under
#                                 ../layers; it exists to keep the bypass
#                                 visible and app-local, never to make it ABI.
CAUSALLM_COMMON_INCLUDES := \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(LOCAL_PATH)/../models \
    $(LOCAL_PATH)/../models/gpt_oss \
    $(LOCAL_PATH)/../models/gpt_oss_cached_slim \
    $(LOCAL_PATH)/../models/qwen2 \
    $(LOCAL_PATH)/../models/qwen3 \
    $(LOCAL_PATH)/../models/qwen3_moe \
    $(LOCAL_PATH)/../models/qwen3_slim_moe \
    $(LOCAL_PATH)/../models/qwen3_cached_slim_moe \
    $(LOCAL_PATH)/../models/gemma2 \
    $(LOCAL_PATH)/../models/gemma3 \
    $(LOCAL_PATH)/../models/bert \
    $(LOCAL_PATH)/../models/timm_vit \
    $(LOCAL_PATH)/../models/deberta_v2 \
    $(LOCAL_PATH)/../models/gemma4 \
    $(LOCAL_PATH)/../models/xlm_roberta \
    $(LOCAL_PATH)/../models/lfm2 \
    $(LOCAL_PATH)/../third_party/minja/include \
    $(LOCAL_PATH)/../third_party \
    $(NNTRAINER_ROOT)/nntrainer/utils \
    $(NNTRAINER_ROOT)/nntrainer/tensor \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cl_operations \

# Common compile flags. -std=c++17/-fexceptions/-frtti come from Application.mk
# (APP_CPPFLAGS); -march and the ABI defines are inherited from the prebuilt
# nntrainer modules below via LOCAL_EXPORT_CFLAGS.
#
# ENABLE_OPENCL is one of those inherited defines and must not be repeated
# here: it selects the app's GPU-routed layer implementations (the cl paths
# inside mha_core) and it also gates the cl_context.h member of engine.h, and
# cl_context.h is installed only when nntrainer itself was configured with
# -Denable-opencl=true. Hardcoding the define makes the app disagree with the
# library it links, which is an ABI break on a CPU-only prebuilt rather than a
# missing feature. Build nntrainer with -Denable-opencl=true instead
# (build_android.sh forwards -D* verbatim) and the define arrives with the
# prebuilt export.
CAUSALLM_COMMON_CFLAGS := -O3 -ffast-math \
    -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator

# Prebuilt nntrainer libraries. The generated Android.mk exports the include
# paths and the -march/FP16 cflags the prebuilts were built with.
NNTRAINER_PREBUILT_MK := $(NNTRAINER_ROOT)/builddir/android_build_result/Android.mk
ifeq ($(wildcard $(NNTRAINER_PREBUILT_MK)),)
$(error $(NNTRAINER_PREBUILT_MK) not found. Build nntrainer first (tools/package_android.sh))
endif
include $(NNTRAINER_PREBUILT_MK)
LOCAL_PATH := $(CAUSALLM_JNI_PATH)

# No OpenCL prebuilt module here: every module below reaches the driver
# through libnntrainer.so, which loads it via its own loader. A
# PREBUILT_SHARED_LIBRARY is validated at parse time, so declaring one for
# builddir/opencl aborts ndk-build outright whenever nntrainer was configured
# without -Denable-opencl=true and that tree was never downloaded.

# Tokenizer library
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := ../lib/libtokenizers_android_c.a
include $(PREBUILT_STATIC_LIBRARY)

# Build libcausallm_core.so (shared library - without api)
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := causallm_core
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := \
    ../chat_template.cpp \
    ../models/causal_lm.cpp \
    ../models/transformer.cpp \
    ../models/sentence_transformer.cpp \
    ../models/model_registry.cpp \
    ../kv_cache_manager.cpp \
    ../models/qwen2/qwen2_causallm.cpp \
    ../models/qwen2/qwen2_embedding.cpp \
    ../models/qwen3/qwen3_causallm.cpp \
    ../models/qwen3/qwen3_embedding.cpp \
    ../models/qwen3_moe/qwen3_moe_causallm.cpp \
    ../models/qwen3_slim_moe/qwen3_slim_moe_causallm.cpp \
    ../models/qwen3_cached_slim_moe/qwen3_cached_slim_moe_causallm.cpp \
    ../models/gpt_oss/gptoss_causallm.cpp \
    ../models/gpt_oss_cached_slim/gptoss_cached_slim_causallm.cpp \
    ../huggingface_tokenizer.cpp \
    ../llm_util.cpp \
    ../layers/embedding_layer.cpp \
    ../layers/embedding_pooling_layer.cpp \
    ../layers/embedding_normalize_layer.cpp \
    ../layers/per_layer_slice.cpp \
    ../layers/mha_core.cpp \
    ../models/qwen3_moe/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/custom_multiply.cpp \
    ../layers/causal_conv1d_layer.cpp \
    ../layers/rms_reverse_norm.cpp \
    ../layers/rms_norm.cpp \
    ../models/qwen3_cached_slim_moe/qwen_moe_layer_cached.cpp \
    ../models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp \
    ../models/gpt_oss/gpt_oss_moe_layer.cpp \
    ../models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.cpp \
    ../models/gemma2/gemma2_causallm.cpp \
    ../models/gemma3/gemma3_causallm.cpp \
    ../models/gemma3/embedding_gemma.cpp \
    ../models/gemma4/gemma4_causallm.cpp \
    ../models/lfm2/lfm2_causallm.cpp \
    ../models/gemma3/function.cpp \
    ../models/timm_vit/timm_vit_transformer.cpp \
    ../models/deberta_v2/deberta_v2.cpp \
    ../models/bert/bert_transformer.cpp \
    ../models/xlm_roberta/xlm_roberta.cpp \
    ../layers/deberta_attention_layer.cpp \
    ../layers/shared_fully_connected_layer.cpp \
    ../api/streamer.cpp \

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(CAUSALLM_COMMON_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

# Build libcausallm_api.so (shared library - api only)
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := causallm_api
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := \
    ../api/causal_lm_api.cpp \
    ../api/model_config.cpp \
    ../api/callback_streamer.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/../api

include $(BUILD_SHARED_LIBRARY)

# Build nntrainer_causallm executable
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := nntrainer_causallm
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := ../main.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(CAUSALLM_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

# Build test_api executable
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := test_api
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := ../api/test_api.cpp

LOCAL_SHARED_LIBRARIES := causallm_api causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/../api

include $(BUILD_EXECUTABLE)


# Build nntr_quantize executable
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := nntr_quantize
LOCAL_LDLIBS := -llog -landroid

# Source files
LOCAL_SRC_FILES := ../quantize.cpp \
    ../models/causal_lm.cpp \
    ../models/transformer.cpp \
    ../models/sentence_transformer.cpp \
    ../models/model_registry.cpp \
    ../kv_cache_manager.cpp \
    ../models/qwen2/qwen2_causallm.cpp \
    ../models/qwen2/qwen2_embedding.cpp \
    ../models/qwen3/qwen3_causallm.cpp \
    ../models/qwen3/qwen3_embedding.cpp \
    ../models/qwen3_moe/qwen3_moe_causallm.cpp \
    ../models/qwen3_slim_moe/qwen3_slim_moe_causallm.cpp \
    ../models/qwen3_cached_slim_moe/qwen3_cached_slim_moe_causallm.cpp \
    ../models/gpt_oss/gptoss_causallm.cpp \
    ../models/gpt_oss_cached_slim/gptoss_cached_slim_causallm.cpp \
    ../huggingface_tokenizer.cpp \
    ../llm_util.cpp \
    ../layers/embedding_layer.cpp \
    ../layers/embedding_pooling_layer.cpp \
    ../layers/embedding_normalize_layer.cpp \
    ../layers/per_layer_slice.cpp \
    ../layers/mha_core.cpp \
    ../models/qwen3_moe/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/custom_multiply.cpp \
    ../layers/causal_conv1d_layer.cpp \
    ../layers/rms_reverse_norm.cpp \
    ../layers/rms_norm.cpp \
    ../models/qwen3_cached_slim_moe/qwen_moe_layer_cached.cpp \
    ../models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp \
    ../models/gpt_oss/gpt_oss_moe_layer.cpp \
    ../models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.cpp \
    ../models/gemma2/gemma2_causallm.cpp \
    ../models/gemma3/gemma3_causallm.cpp \
    ../models/gemma3/embedding_gemma.cpp \
    ../models/gemma4/gemma4_causallm.cpp \
    ../models/lfm2/lfm2_causallm.cpp \
    ../models/gemma3/function.cpp \
    ../models/deberta_v2/deberta_v2.cpp \
    ../models/bert/bert_transformer.cpp \
    ../models/xlm_roberta/xlm_roberta.cpp \
    ../layers/deberta_attention_layer.cpp \
    ../layers/shared_fully_connected_layer.cpp \
    ../api/streamer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(LOCAL_PATH)/../models \
    $(LOCAL_PATH)/../models/gpt_oss \
    $(LOCAL_PATH)/../models/gpt_oss_cached_slim \
    $(LOCAL_PATH)/../models/qwen2 \
    $(LOCAL_PATH)/../models/qwen3 \
    $(LOCAL_PATH)/../models/qwen3_moe \
    $(LOCAL_PATH)/../models/qwen3_slim_moe \
    $(LOCAL_PATH)/../models/qwen3_cached_slim_moe \
    $(LOCAL_PATH)/../models/gemma2 \
    $(LOCAL_PATH)/../models/gemma3 \
    $(LOCAL_PATH)/../models/bert \
    $(LOCAL_PATH)/../models/deberta_v2 \
    $(LOCAL_PATH)/../models/gemma4 \
    $(LOCAL_PATH)/../models/xlm_roberta \
    $(LOCAL_PATH)/../models/lfm2 \
    $(NNTRAINER_ROOT)/nntrainer/utils \
    $(NNTRAINER_ROOT)/nntrainer/tensor \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cl_operations \

include $(BUILD_EXECUTABLE)

# Build nntr_safetensors_info executable
include $(CLEAR_VARS)

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS)
LOCAL_MODULE := nntr_safetensors_info
LOCAL_LDLIBS := -llog -landroid

# Source files (header-only inspector; uses safetensors_util from libnntrainer)
LOCAL_SRC_FILES := ../safetensors_info.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(LOCAL_PATH)/..

include $(BUILD_EXECUTABLE)

# ---- googletest (vendored from $ANDROID_NDK/sources/third_party/googletest) ----
# Mirrors the pattern used by test/jni/Android.mk so the CausalLM unit tests can
# be cross-compiled and run on-device via adb.
include $(CLEAR_VARS)
GTEST_PATH := googletest
LOCAL_MODULE := googletest_main
LOCAL_CPP_FEATURES := rtti exceptions
LOCAL_C_INCLUDES := $(LOCAL_PATH)/$(GTEST_PATH)/include $(LOCAL_PATH)/$(GTEST_PATH)
LOCAL_CFLAGS := -std=c++17 -frtti -fexceptions
LOCAL_SRC_FILES := \
    $(GTEST_PATH)/src/gtest-all.cc \
    $(GTEST_PATH)/src/gtest_main.cc
include $(BUILD_STATIC_LIBRARY)

# ---- unittest_causallm_models (CausalLM reference/differential gtest suite) ----
# Builds the recently-added differential tests (causallm_test_utils.cpp + every
# unittest_causallm_*.cpp listed in Applications/CausalLM/meson.build). Built
# with the same FP16 ABI flags as causallm_core so the prebuilt shared libs link.
include $(CLEAR_VARS)

LOCAL_MODULE := unittest_causallm_models

LOCAL_CFLAGS += $(CAUSALLM_COMMON_CFLAGS) -Igoogletest/include -Igoogletest/
LOCAL_LDLIBS := -llog -landroid

UNITTEST_MODELS_DIR := ../../../test/unittest/models
LOCAL_SRC_FILES := \
    $(UNITTEST_MODELS_DIR)/causallm_test_utils.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gemma3.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gemma3_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gemma4.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gemma4_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_moe.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_moe_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_slim_moe.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_cached_slim_moe.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gpt_oss.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_gpt_oss_cached_slim.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen2.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen2_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen3_embedding_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_qwen2_embedding_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_embedding_gemma_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_tinybert_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_deberta_v2_reference.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_lfm2.cpp \
    $(UNITTEST_MODELS_DIR)/unittest_causallm_lfm2_reference.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main

LOCAL_C_INCLUDES += $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/$(GTEST_PATH)/include \
    $(LOCAL_PATH)/../api \
    $(LOCAL_PATH)/$(UNITTEST_MODELS_DIR)

include $(BUILD_EXECUTABLE)
