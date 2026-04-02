LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../../
endif

NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	$(NNTRAINER_ROOT)/api/capi/include

NNTRAINER_APPLICATION := $(NNTRAINER_ROOT)/Applications

ML_API_COMMON_ROOT := ${NNTRAINER_ROOT}/ml_api_common

include $(CLEAR_VARS)

LOCAL_MODULE := gstreamer
LOCAL_SRC_FILES := ${ML_API_COMMON_ROOT}/lib/arm64-v8a/libgstreamer_android.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ml-api-inference
LOCAL_SRC_FILES :=  ${ML_API_COMMON_ROOT}/lib/arm64-v8a/libnnstreamer-native.so
LOCAL_SHARED_LIBRARIES := gstreamer
LOCAL_EXPORT_C_INCLUDES := $(ML_API_COMMON_ROOT)/include
LOCAL_EXPORT_CFLAGS += -DUSE_BLAS=1

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := app_utils
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/Applications/utils/libs/$(TARGET_ARCH_ABI)/libapp_utils.so
APP_UTILS_INCLUDES := $(NNTRAINER_ROOT)/Applications/utils/jni/includes

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
LOCAL_SHARED_LIBRARIES := nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := capi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libcapi-nntrainer.so
LOCAL_SHARED_LIBRARIES := ccapi-nntrainer ml-api-inference

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
LOCAL_MODULE := nntrainer_training
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := main.cpp

# @todo add api for context
LOCAL_SHARED_LIBRARIES := nntrainer capi-nntrainer app_utils

LOCAL_C_INCLUDES += $(TFLITE_INCLUDES) $(NNTRAINER_INCLUDES) $(APP_UTILS_INCLUDES)

include $(BUILD_EXECUTABLE)
