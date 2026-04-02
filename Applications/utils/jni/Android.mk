# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j2
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

NNTRAINER_APPLICATION := $(NNTRAINER_ROOT)/Applications

UTILS_SRCS := $(NNTRAINER_APPLICATION)/utils/jni/bitmap_helpers.cpp

UTILS_INCLUDES := $(NNTRAINER_APPLICATION)/utils/jni/includes

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -pthread -fexceptions
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions
LOCAL_MODULE_TAGS   := optional

LOCAL_LDLIBS        := -llog -landroid

LOCAL_MODULE        := app_utils
LOCAL_SRC_FILES     := $(UTILS_SRCS)
LOCAL_C_INCLUDES    += $(UTILS_INCLUDES)

include $(BUILD_SHARED_LIBRARY)
