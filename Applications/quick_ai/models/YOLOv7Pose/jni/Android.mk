LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../../../..
endif

# Prebuilt nntrainer shared libs (produced by the meson-android build under
# build_android/jni/arm64-v8a). Override NNTRAINER_LIBS if they live elsewhere.
ifndef NNTRAINER_LIBS
NNTRAINER_LIBS := $(NNTRAINER_ROOT)/build_android/jni/$(TARGET_ARCH_ABI)
endif

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	$(ML_API_COMMON_INCLUDES)

# quick_ai custom layers used by the pose head.
QUICK_AI_LAYERS := $(NNTRAINER_ROOT)/Applications/quick_ai/layers

include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE := yolov7_pose_infer
LOCAL_MODULE_TAGS := optional
LOCAL_LDLIBS := -llog -landroid

# Match the library's ISA features so int8 (dotprod/i8mm) and fp16 paths
# compile consistently with libnntrainer.
POSE_CFLAGS := -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -pthread \
	-fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_CFLAGS += $(POSE_CFLAGS)
LOCAL_CXXFLAGS += $(POSE_CFLAGS)
LOCAL_LDFLAGS += -fexceptions

# The custom rtmcc_head layer is compiled straight into the executable
# (registered at runtime in main.cpp).
LOCAL_SRC_FILES := main.cpp \
	$(QUICK_AI_LAYERS)/rtmcc_head.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(QUICK_AI_LAYERS) \
	$(LOCAL_PATH)/../../../third_party

include $(BUILD_EXECUTABLE)
