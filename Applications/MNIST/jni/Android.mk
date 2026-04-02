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
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/include \
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
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions
LOCAL_LDFLAGS += -fexceptions
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_mnist
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := main.cpp

LOCAL_SHARED_LIBRARIES := ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)
