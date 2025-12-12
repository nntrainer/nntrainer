LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ML_API_COMMON_INCLUDES := ${NNTRAINER_ROOT}/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/builddir/android_build_result/include/nntrainer 

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

# Build CausalLM executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_causallm
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

# Source files
LOCAL_SRC_FILES := ../main.cpp \
    ../models/causal_lm.cpp \
    ../models/qwen3/qwen3_causallm.cpp \
    ../models/qwen3_moe/qwen3_moe_causallm.cpp \
    ../models/qwen3_slim_moe/qwen3_slim_moe_causallm.cpp \
    ../models/qwen3_cached_slim_moe/qwen3_cached_slim_moe_causallm.cpp \
    ../models/gpt_oss/gptoss_causallm.cpp \
    ../models/gpt_oss_cached_slim/gptoss_cached_slim_causallm.cpp \
    ../huggingface_tokenizer.cpp \
    ../llm_util.cpp \
    ../layers/embedding_layer.cpp \
    ../layers/mha_core.cpp \
    ../models/qwen3_moe/qwen_moe_layer.cpp \
    ../layers/reshaped_rms_norm.cpp \
    ../layers/rms_norm.cpp \
    ../layers/swiglu.cpp \
    ../layers/tie_word_embedding.cpp\
    ../models/qwen3_cached_slim_moe/qwen_moe_layer_cached.cpp \
    ../layers/qkv_layer.cpp \
    ../models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp \
    ../models/gpt_oss/gpt_oss_moe_layer.cpp \
    ../models/gpt_oss_cached_slim/gpt_oss_moe_layer_cached.cpp \

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
    $(LOCAL_PATH)/.. \
    $(LOCAL_PATH)/../layers \
    $(LOCAL_PATH)/../models \
    $(LOCAL_PATH)/../models/gpt_oss \
    $(LOCAL_PATH)/../models/gpt_oss_cached_slim \
    $(LOCAL_PATH)/../models/qwen3 \
    $(LOCAL_PATH)/../models/qwen3_moe \
    $(LOCAL_PATH)/../models/qwen3_slim_moe \
    $(LOCAL_PATH)/../models/qwen3_cached_slim_moe \

include $(BUILD_EXECUTABLE)
