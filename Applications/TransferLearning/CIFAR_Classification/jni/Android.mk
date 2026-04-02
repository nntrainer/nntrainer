LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../..
endif

 ML_API_COMMON_INCLUDES := ${NNTRAINER_ROOT}/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	${ML_API_COMMON_INCLUDES}

NNTRAINER_APPLICATION := $(NNTRAINER_ROOT)/Applications

include $(CLEAR_VARS)

TENSORFLOW_VERSION := 2.3.0

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

$(info $(shell ($(NNTRAINER_APPLICATION)/prepare_tflite.sh $(TENSORFLOW_VERSION) $(NNTRAINER_APPLICATION))))

TENSORFLOW_ROOT := $(NNTRAINER_APPLICATION)/tensorflow-$(TENSORFLOW_VERSION)/tensorflow-lite

endif
endif

LOCAL_MODULE := tensorflow-lite
LIB_ := arm64

ifeq ($(APP_ABI), armeabi-v7a)
	LIB_ := armv7
endif
LOCAL_SRC_FILES := $(TENSORFLOW_ROOT)/lib/$(LIB_)/libtensorflow-lite.a
LOCAL_EXPORT_C_INCLUDES := $(TENSORFLOW_ROOT)/include
LOCAL_EXPORT_LDLIBS := -lEGL -lGLESv2

include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := app_utils
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/Applications/utils/libs/$(TARGET_ARCH_ABI)/libapp_utils.so
APP_UTILS_INCLUDES := $(NNTRAINER_ROOT)/Applications/utils/jni/includes

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
LOCAL_MODULE := nntrainer_classification
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := main.cpp

# migrate this to ccapi
LOCAL_SHARED_LIBRARIES := nntrainer app_utils

LOCAL_STATIC_LIBRARIES := tensorflow-lite

LOCAL_C_INCLUDES += $(TFLITE_INCLUDES) $(NNTRAINER_INCLUDES) $(APP_UTILS_INCLUDES)

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions
LOCAL_LDFLAGS += -fexceptions
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_classification_func
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := main_func.cpp

LOCAL_SHARED_LIBRARIES := ccapi-nntrainer app_utils
# nntrainer is added for app_context, this should be removed
LOCAL_SHARED_LIBRARIES += nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(APP_UTILS_INCLUDES)

include $(BUILD_EXECUTABLE)
