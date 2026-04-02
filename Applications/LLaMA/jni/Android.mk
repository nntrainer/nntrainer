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
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	${ML_API_COMMON_INCLUDES}

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := rms_norm_layer
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1

LOCAL_SRC_FILES := rms_norm.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := swiglu_layer
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1

LOCAL_SRC_FILES := swiglu.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := custom_multi_head_attention_layer
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1

LOCAL_SRC_FILES := custom_multi_head_attention_layer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_SHARED_LIBRARY)


include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_LDFLAGS += -fexceptions -DENABLE_FP16=1 -DUSE__FP16=1
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_llama
LOCAL_LDLIBS := -llog -landroid -DENABLE_FP16=1 -DUSE__FP16=1

LOCAL_SRC_FILES := main.cpp 

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer rms_norm_layer swiglu_layer custom_multi_head_attention_layer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)
