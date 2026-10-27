LOCAL_PATH := $(call my-dir)

NNTRAINER_ROOT := $(LOCAL_PATH)/../../../../..
NNTRAINER_LIBS := $(NNTRAINER_ROOT)/build_android/jni/arm64-v8a
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/api $(NNTRAINER_ROOT)/api/ccapi/include $(NNTRAINER_ROOT)/nntrainer/utils $(NNTRAINER_ROOT)/nntrainer/tensor $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm $(NNTRAINER_ROOT)/nntrainer/layers $(NNTRAINER_ROOT)/nntrainer/graph $(NNTRAINER_ROOT)/nntrainer/optimizers $(NNTRAINER_ROOT)/nntrainer

include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_LIBS)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := fastvit_keyword_infer

LOCAL_CFLAGS += -std=c++17 -O3 -fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_CXXFLAGS += -std=c++17 -O3 -fexceptions -frtti -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1
LOCAL_LDFLAGS += -fexceptions
LOCAL_LDLIBS += -llog

LOCAL_SRC_FILES := main.cpp fastvit_attention_layer.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) $(LOCAL_PATH)/../../../third_party

include $(BUILD_EXECUTABLE)
