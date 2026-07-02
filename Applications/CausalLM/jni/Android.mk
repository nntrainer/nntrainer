LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/builddir/android_build_result/include/nntrainer

# Common Includes Definition
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
    $(LOCAL_PATH)/../models/gemma3 \
    $(LOCAL_PATH)/../models/timm_vit \
    $(LOCAL_PATH)/../models/deberta_v2 \
    $(LOCAL_PATH)/../models/gemma4 \
    $(LOCAL_PATH)/../third_party/minja/include \
    $(LOCAL_PATH)/../third_party \

# Prebuilt nntrainer libraries
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

# Tokenizer library
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := ../lib/libtokenizers_android_c.a
include $(PREBUILT_STATIC_LIBRARY)

# Build libcausallm_core.so (shared library - without api)
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_ARM_MODE := arm
LOCAL_MODULE := causallm_core
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

LOCAL_SRC_FILES := \
    ../chat_template.cpp \
    ../models/causal_lm.cpp \
    ../models/transformer.cpp \
    ../models/sentence_transformer.cpp \
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
    ../layers/scalar_multiply.cpp \
    ../layers/logit_softcapping.cpp \
    ../layers/mha_core.cpp \
    ../layers/lm_head.cpp \
    ../models/qwen3_moe/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/rms_norm.cpp \
    ../layers/swiglu.cpp \
    ../layers/tie_word_embedding.cpp \
    ../models/qwen3_cached_slim_moe/qwen_moe_layer_cached.cpp \
    ../layers/qkv_layer.cpp \
    ../models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp \
    ../models/gpt_oss/gpt_oss_moe_layer.cpp \
    ../models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.cpp \
    ../models/gemma3/gemma3_causallm.cpp \
    ../models/gemma3/embedding_gemma.cpp \
    ../models/gemma4/gemma4_causallm.cpp \
    ../models/gemma3/function.cpp \
    ../models/timm_vit/timm_vit_transformer.cpp \
    ../models/deberta_v2/deberta_v2.cpp \
    ../layers/deberta_attention_layer.cpp \
    ../layers/shared_fully_connected_layer.cpp \
    ../api/streamer.cpp \

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_COMMON_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

# Build libcausallm_api.so (shared library - api only)
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_ARM_MODE := arm
LOCAL_MODULE := causallm_api
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

LOCAL_SRC_FILES := \
    ../api/causal_lm_api.cpp \
    ../api/model_config.cpp \
    ../api/callback_streamer.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/../api

include $(BUILD_SHARED_LIBRARY)

# Build nntrainer_causallm executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_causallm
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

LOCAL_SRC_FILES := ../main.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_COMMON_INCLUDES)

include $(BUILD_EXECUTABLE)

# Build test_api executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := test_api
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

LOCAL_SRC_FILES := ../api/test_api.cpp

LOCAL_SHARED_LIBRARIES := causallm_api causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/../api

include $(BUILD_EXECUTABLE)


# Build nntr_quantize executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntr_quantize
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

# Source files
LOCAL_SRC_FILES := ../quantize.cpp \
    ../models/causal_lm.cpp \
    ../models/transformer.cpp \
    ../models/sentence_transformer.cpp \
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
    ../llm_util.cpp \
    ../layers/embedding_layer.cpp \
    ../layers/embedding_pooling_layer.cpp \
    ../layers/embedding_normalize_layer.cpp \
    ../layers/per_layer_slice.cpp \
    ../layers/scalar_multiply.cpp \
    ../layers/logit_softcapping.cpp \
    ../layers/mha_core.cpp \
    ../models/qwen3_moe/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/rms_norm.cpp \
    ../layers/swiglu.cpp \
    ../layers/tie_word_embedding.cpp\
    ../layers/lm_head.cpp\
    ../models/qwen3_cached_slim_moe/qwen_moe_layer_cached.cpp \
    ../layers/qkv_layer.cpp \
    ../models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp \
    ../models/gpt_oss/gpt_oss_moe_layer.cpp \
    ../models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.cpp \
    ../models/gemma3/gemma3_causallm.cpp \
    ../models/gemma3/embedding_gemma.cpp \
    ../models/gemma4/gemma4_causallm.cpp \
    ../models/gemma3/function.cpp \
    ../models/deberta_v2/deberta_v2.cpp \
    ../layers/deberta_attention_layer.cpp \
    ../layers/shared_fully_connected_layer.cpp \
    ../api/streamer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
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
    $(LOCAL_PATH)/../models/gemma3 \
    $(LOCAL_PATH)/../models/deberta_v2 \
    $(LOCAL_PATH)/../models/gemma4 \

include $(BUILD_EXECUTABLE)

# Build nntr_safetensors_info executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntr_safetensors_info
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

# Source files (header-only inspector; uses safetensors_util from libnntrainer)
LOCAL_SRC_FILES := ../safetensors_info.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
    $(LOCAL_PATH)/..

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

LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE_TAGS := optional
LOCAL_MODULE := unittest_causallm_models

CAUSALLM_TEST_FLAGS := -pthread -fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_CFLAGS += -std=c++17 $(CAUSALLM_TEST_FLAGS) -Igoogletest/include -Igoogletest/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += -fexceptions
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
    $(UNITTEST_MODELS_DIR)/unittest_causallm_deberta_v2_reference.cpp

LOCAL_SHARED_LIBRARIES := causallm_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := googletest_main

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CAUSALLM_COMMON_INCLUDES) \
    $(LOCAL_PATH)/$(GTEST_PATH)/include \
    $(LOCAL_PATH)/../api \
    $(LOCAL_PATH)/$(UNITTEST_MODELS_DIR)

include $(BUILD_EXECUTABLE)
