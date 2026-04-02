LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(LOCAL_PATH)/nntrainer/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/nntrainer/include/nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(LOCAL_PATH)/nntrainer/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/nntrainer/include/nntrainer

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

CIFARDIR = ../../../../../../utils/datagen/cifar

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti -fexceptions
LOCAL_CFLAGS += -pthread -fexceptions
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntrainer_resnet
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := NNTrainer.cpp $(CIFARDIR)/cifar_dataloader.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(CIFARDIR)

include $(BUILD_SHARED_LIBRARY)
